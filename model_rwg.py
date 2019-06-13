import time
from embeddings import *
from constants import *
from tqdm import tqdm
from utils.model_utils import model_checkpoint, parallel, device
from utils.misc_utils import write_line_to_file, label_tsr_to_one_hot_tsr


class RelevantWordGenerator(nn.Module):

    def __init__(self, src_embed, encoder, aggr, word_generator, dropout_prob=0.0):
        super(RelevantWordGenerator, self).__init__()
        self.src_embed = parallel(src_embed)
        self.encoder = parallel(encoder)
        self.aggr = aggr
        self.word_generator = parallel(word_generator)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, data_dict):
        src = data_dict[DK_SRC_WID].to(device())
        src_oov = data_dict[DK_SRC_OOV_WID].to(device())
        src = self.src_embed(src, src_oov)
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        encoder_op = self.encoder(src, mask=src_mask)
        encoder_op = self.dropout(encoder_op)
        if hasattr(self.aggr, "require_mask"):
            aggr_op = self.aggr(encoder_op, mask=src_mask)
        else:
            aggr_op = self.aggr(encoder_op)
        word_probs = self.word_generator(aggr_op)
        return word_probs


class MeanMaxOutAggr(nn.Module):

    def __init__(self, max_out):
        super(MeanMaxOutAggr, self).__init__()
        self.max_out = max_out

    def forward(self, x, dim=1):
        if x.size(dim) == 1:
            rv = x
        else:
            rv = x.mean(dim=dim, keepdim=True)
        return self.max_out(rv)


class MaxOut(nn.Module):

    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
        if self.pool_size <= 1:
            return input
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)
        return output

    def __repr__(self):
        return "MaxOut(pool_size={})".format(self.pool_size)

    def __str__(self):
        return "MaxOut(pool_size={})".format(self.pool_size)


class SimpleRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1, rnn_dir=2, dropout_prob=0.0, rnn_type="gru",
                 return_aggr_vector_only=False, return_output_vector_only=False,
                 output_resize_layer=None, avoid_parallel=True):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dir = rnn_dir
        self.rnn_type = rnn_type
        self.dropout_prob = dropout_prob
        self.return_aggr_vector_only = return_aggr_vector_only
        self.return_output_vector_only = return_output_vector_only
        self.output_resize_layer = output_resize_layer
        self.avoid_parallel = avoid_parallel
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=self.rnn_dir==2)
        else:
            self.rnn = nn.GRU(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=self.rnn_dir==2)
        init_weights(self, base_model_type="rnn")

    def forward(self, embedded, hidden=None, lens=None, mask=None):
        if mask is not None:
            lens = mask.sum(dim=-1).squeeze()
        self.rnn.flatten_parameters()
        if lens is None:
            outputs, hidden = self.rnn(embedded, hidden)
            rv_lens = None
        else:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lens, batch_first=True)
            outputs, hidden = self.rnn(packed, hidden)
            outputs, rv_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        if self.output_resize_layer is not None:
            outputs = self.output_resize_layer(outputs)

        if self.return_aggr_vector_only:
            if rv_lens is not None:
                rv_lens = rv_lens.to(device())
                rv_lens = rv_lens-1
                rv = torch.gather(outputs, 1, rv_lens.view(-1, 1).unsqueeze(2).repeat(1, 1, outputs.size(-1)))
            else:
                rv = outputs
            return rv
        elif self.return_output_vector_only:
            return outputs
        else:
            return outputs, hidden


class Generator(nn.Module):

    def __init__(self, d_model, vocab, use_log_softmax=True):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.use_log_softmax = use_log_softmax
        init_weights(self)

    def forward(self, x):
        logits = self.proj(x)
        if self.use_log_softmax:
            probs = F.log_softmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs


def make_rwg_model(src_w2v_mat, params, src_vocab_size, tgt_vocab_size, src_oov_vocab_size,
                   use_log_softmax=True):
    d_model = params.rwg_hidden_size
    dropout_prob = params.rwg_dropout_prob
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=True)
    src_embed_layer = src_word_embed_layer
    encoder = SimpleRNN(params.word_embedding_dim, d_model, return_aggr_vector_only=True,
                        rnn_dir=params.rwg_rnn_dir, n_layers=params.rwg_rnn_layers)
    pool_size = params.rwg_rnn_pool_size
    d_input_gen = params.rwg_rnn_dir * d_model // pool_size
    aggr = MeanMaxOutAggr(MaxOut(pool_size=pool_size))
    word_generator = Generator(d_input_gen, tgt_vocab_size, use_log_softmax=use_log_softmax)
    model = RelevantWordGenerator(
        src_embed_layer,
        encoder,
        aggr,
        word_generator,
        dropout_prob
    )
    return model


def train_rwg(params, model, train_loader, criterion, optimizer,
              completed_epochs=0, eval_loader=None, best_eval_result=0,
              best_eval_epoch=0, past_eval_results=[], past_train_loss=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_rwg_epoch(train_loader, model, criterion, optimizer,
                                      model_name=params.model_name, report_acc=False,
                                      max_grad_norm=params.max_gradient_norm, pad_idx=params.pad_idx,
                                      curr_epoch=report_epoch, logs_dir=params.logs_dir)

        past_train_loss.append(train_loss)
        if len(past_train_loss) > 2: past_train_loss = past_train_loss[1:]
        if params.lr_decay_with_train_perf:
            if params.lr_decay_with_train_loss_diff and \
                len(past_train_loss) == 2 and \
                past_train_loss[1] <= past_train_loss[0] and \
                past_train_loss[0] - past_train_loss[1] < params.train_loss_diff_threshold:
                print("updating lr by train loss diff, threshold: {}".format(params.train_loss_diff_threshold))
                optimizer.shrink_learning_rate()
                past_train_loss = []
            elif hasattr(optimizer, "update_learning_rate"):
                optimizer.update_learning_rate(train_loss, "max")

        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        model_checkpoint(fn, report_epoch, model, optimizer, params,
                         past_eval_results, best_eval_result, best_eval_epoch)

        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0:
                    eval_loss, eval_score = run_rwg_epoch(eval_loader, model, criterion, None,
                                                          pad_idx=params.pad_idx, model_name=params.model_name,
                                                          report_acc=True, curr_epoch=report_epoch,
                                                          logs_dir=None, desc="Eval")
                    if eval_score > best_eval_result:
                        best_eval_result = eval_score
                        best_eval_epoch = report_epoch
                        print("Model best checkpoint with score {}".format(eval_score))
                        fn = params.saved_models_dir + params.model_name + "_best.pt"
                        model_checkpoint(fn, report_epoch, model, optimizer, params,
                                         past_eval_results, best_eval_result, best_eval_epoch)
                    info = "Best {} so far {} from epoch {}".format(params.eval_metric, best_eval_result, best_eval_epoch)
                    print(info)
                    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
                    if hasattr(optimizer, "update_learning_rate") and not params.lr_decay_with_train_perf:
                        optimizer.update_learning_rate(eval_score, "min")
                    past_eval_results.append(eval_score)
                    if len(past_eval_results) > params.past_eval_scores_considered:
                        past_eval_results = past_eval_results[1:]

        print("")


def run_rwg_epoch(data_iter, model, criterion, optimizer,
                  model_name="rwg", desc="Train", curr_epoch=0, pad_idx=0,
                  logs_dir=None, max_grad_norm=5.0, report_acc=False):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct_k = 0
    total_correct_2k = 0
    total_correct_10 = 0
    total_correct_25 = 0
    total_correct_50 = 0
    total_correct_100 = 0
    total_correct_500 = 0
    total_acc_tokens = 0
    for batch in tqdm(data_iter, mininterval=2, desc=desc, leave=False, ascii=True):
        probs = model.forward(batch)
        gen_targets = batch[DK_TGT_GEN_WID]
        gen_targets = label_tsr_to_one_hot_tsr(gen_targets, probs.size(-1))
        gen_targets = gen_targets.to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        probs = probs.view(-1, probs.size(-1))
        loss = criterion(probs, gen_targets.contiguous())
        total_loss += loss.item()
        total_tokens += n_tokens
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optimizer.step()
            if torch.isnan(loss).any():
                assert False, "nan detected after step()"

        if report_acc:
            gen_targets = batch[DK_TGT_GEN_WID]
            for bi in range(batch[DK_BATCH_SIZE]):
                k = batch[DK_WI_N_WORDS][bi]
                prob = probs[bi,:].squeeze()
                _, top_k_ids = prob.topk(500)
                pred_k_ids = top_k_ids.tolist()
                pred_tk_ids = set(pred_k_ids[:k])
                pred_2k_ids = set(pred_k_ids[:int(2*k)])
                pred_10_ids = set(pred_k_ids[:10])
                pred_25_ids = set(pred_k_ids[:25])
                pred_50_ids = set(pred_k_ids[:50])
                pred_100_ids = set(pred_k_ids[:100])
                pred_500_ids = set(pred_k_ids)
                truth_ids = gen_targets[bi,:].squeeze()
                for truth_id in truth_ids.tolist():
                    if truth_id == pad_idx: continue
                    if truth_id in pred_tk_ids:
                        total_correct_k += 1
                    if truth_id in pred_2k_ids:
                        total_correct_2k += 1
                    if truth_id in pred_10_ids:
                        total_correct_10 += 1
                    if truth_id in pred_25_ids:
                        total_correct_25 += 1
                    if truth_id in pred_50_ids:
                        total_correct_50 += 1
                    if truth_id in pred_100_ids:
                        total_correct_100 += 1
                    if truth_id in pred_500_ids:
                        total_correct_500 += 1
                    total_acc_tokens += 1

    elapsed = time.time() - start
    if report_acc:
        info = desc + " epoch %d loss %f top_k acc %f top_2k acc %f top_10 acc %f top_25 acc %f top_50 acc %f top_100 " \
                      "acc %f top_500 acc % f ppl %f elapsed time %f" % (
                curr_epoch, total_loss / total_tokens,
                total_correct_k / total_acc_tokens,
                total_correct_2k / total_acc_tokens,
                total_correct_10 / total_acc_tokens,
                total_correct_25 / total_acc_tokens,
                total_correct_50 / total_acc_tokens,
                total_correct_100 / total_acc_tokens,
                total_correct_500 / total_acc_tokens,
                math.exp(total_loss / total_tokens),
                elapsed)
    else:
        info = desc + " epoch %d loss %f ppl %f elapsed time %f" % (
            curr_epoch, total_loss / total_tokens,
            math.exp(total_loss / total_tokens),
            elapsed)
    print(info)
    if logs_dir is not None:
        write_line_to_file(info, logs_dir + model_name + "_train_info.txt")
    rv_loss = total_loss / total_tokens
    rv_perf = total_correct_100 / total_acc_tokens if total_acc_tokens > 0 else 0
    return rv_loss, rv_perf
