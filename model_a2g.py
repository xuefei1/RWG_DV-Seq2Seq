import time
from tqdm import tqdm
from utils.model_utils import model_checkpoint, parallel
from utils.misc_utils import write_line_to_file, label_tsr_to_one_hot_tsr
from embeddings import *
from components import *


class Attn2Gen(nn.Module):

    def __init__(self, src_embed, encoder, aggr, word_generator, dropout_prob=0.0):
        super(Attn2Gen, self).__init__()
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


class DummyEncoder(nn.Module):

    def __init__(self):
        super(DummyEncoder, self).__init__()
        self.avoid_parallel = True

    def forward(self, x, mask=None):
        return x


# a2g
def make_a2g_model(src_w2v_mat, params, src_vocab_size, tgt_vocab_size, src_oov_vocab_size,
                   use_log_softmax=True):
    d_model = params.a2g_hidden_size
    n_heads = params.a2g_num_attn_heads
    dropout_prob = params.a2g_dropout_prob
    ff = PositionwiseFeedForward(d_model, params.a2g_feedforward_hidden_size)
    attn = MultiHeadedAttention(n_heads, d_model)
    pos_encode = PositionalEncoding(d_model)
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=True)
        # src_word_embed_layer = PartiallyTrainableEmbedding(params.word_embedding_dim, src_w2v_mat, src_oov_vocab_size,
        #                                                    padding_idx=params.pad_idx)
    src_embed_layer = ResizeWrapperEmbedding(d_model, src_word_embed_layer, multiply_by_sqrt_d_model=True)
    # src_embed_layer = src_word_embed_layer

    enc_layer = EncoderLayer(d_model, attn, ff, dropout_prob)
    encoder = PosEncoder(pos_encode, enc_layer, params.a2g_num_encoder_layers)

    # encoder = SimpleRNN(params.word_embedding_dim, d_model,
    #                     return_aggr_vector_only=True, rnn_dir=params.a2g_rnn_dir, n_layers=params.a2g_rnn_layers)
    # pool_size = params.a2g_rnn_pool_size
    # d_input_gen = params.a2g_rnn_dir * d_model // pool_size
    # aggr = MeanFFAggr(params.a2g_rnn_dir * d_model, d_input_gen)
    # aggr = MeanMaxOutAggr(MaxOut(pool_size=pool_size))
    # encoder = DummyEncoder()
    aggr = MaskedMeanAggr()
    # aggr = MeanAggr()
    # aggr = MaskSelectAggr()
    # word_generator = FishTailGenerator(d_input_gen, d_input_gen // pool_size, tgt_vocab_size)
    word_generator = Generator(d_model, tgt_vocab_size, use_log_softmax=use_log_softmax)

    # encoder = DummyEncoder()

    # aggr = SimpleRNN(d_model, d_model, return_aggr_vector_only=True,
    #                  output_resize_layer=nn.Linear(2*d_model, d_model, bias=False))
    # aggr = MaskedMeanAggr()
    # aggr = MaxAggr()
    # aggr = nn.Sequential(
    #     MaskedMeanAggr(),
    #     MaxOut(pool_size=2),
    # )
    # word_generator = Generator(d_input_gen, tgt_vocab_size)

    model = Attn2Gen(
        src_embed_layer,
        encoder,
        aggr,
        word_generator,
        dropout_prob
    )
    return model


def make_rnn_based_a2g_model(src_w2v_mat, params, src_vocab_size, tgt_vocab_size, src_oov_vocab_size,
                             use_log_softmax=True):
    d_model = params.a2g_hidden_size
    dropout_prob = params.a2g_dropout_prob
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=True)
        # src_word_embed_layer = PartiallyTrainableEmbedding(params.word_embedding_dim, src_w2v_mat, src_oov_vocab_size,
        #                                                    padding_idx=params.pad_idx)
    src_embed_layer = src_word_embed_layer
    encoder = SimpleRNN(params.word_embedding_dim, d_model, return_aggr_vector_only=True,
                        rnn_dir=params.a2g_rnn_dir, n_layers=params.a2g_rnn_layers)
    pool_size = params.a2g_rnn_pool_size
    d_input_gen = params.a2g_rnn_dir * d_model // pool_size
    # aggr = MeanFFAggr(params.a2g_rnn_dir * d_model, d_input_gen)
    aggr = MeanMaxOutAggr(MaxOut(pool_size=pool_size))
    # aggr = MeanAggr()

    word_generator = Generator(d_input_gen, tgt_vocab_size, use_log_softmax=use_log_softmax)
    model = Attn2Gen(
        src_embed_layer,
        encoder,
        aggr,
        word_generator,
        dropout_prob
    )
    return model


def eval_a2g(model, loader, params, i2w, criterion, k=10, desc="Eval"):
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    ofn = params.logs_dir + params.model_name + "_"+desc.lower()+"_out.txt"
    total_loss = 0
    write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        g_log_wid_probs = model.forward(batch)
        gen_targets = batch[DK_TGT_GEN_WID].to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        g_log_wid_probs = g_log_wid_probs.view(-1, g_log_wid_probs.size(-1))
        loss = criterion(g_log_wid_probs, gen_targets.contiguous().view(-1))
        for bi in range(batch[DK_BATCH_SIZE]):
            prob = g_log_wid_probs[bi, :].squeeze()
            _, top_ids = prob.topk(k)
            pred_ids = top_ids.tolist()
            truth_id = gen_targets[bi, :].squeeze().item()
            pred_words = [i2w[i] for i in pred_ids]
            truth_word = i2w[truth_id]
            write_line_to_file("truth: " + truth_word, ofn)
            write_line_to_file("preds top {}: {}".format( k, " ".join([w for w in pred_words if w not in exclude_tokens])), ofn)
        total_loss += loss.item() / n_tokens
    info = "eval loss {}".format(total_loss/len(loader))
    write_line_to_file(info, ofn)


def eval_a2g_bce(model, loader, params, i2w, criterion, k=10, desc="Eval"):
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    ofn = params.logs_dir + params.model_name + "_"+desc.lower()+"_out.txt"
    total_loss = 0
    write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        probs = model.forward(batch)
        gen_targets = batch[DK_TGT_GEN_WID]
        gen_targets = label_tsr_to_one_hot_tsr(gen_targets, probs.size(-1))
        gen_targets = gen_targets.to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        probs = probs.view(-1, probs.size(-1))
        loss = criterion(probs, gen_targets.contiguous())
        for bi in range(batch[DK_BATCH_SIZE]):
            prob = probs[bi, :].squeeze()
            _, top_ids = prob.topk(k)
            pred_ids = top_ids.tolist()
            truth_id = gen_targets[bi, :].squeeze().item()
            pred_words = [i2w[i] for i in pred_ids]
            truth_word = i2w[truth_id]
            write_line_to_file("truth: " + truth_word, ofn)
            write_line_to_file("preds top {}: {}".format( k, " ".join([w for w in pred_words if w not in exclude_tokens])), ofn)
        total_loss += loss.item() / n_tokens
    info = "eval loss {}".format(total_loss/len(loader))
    write_line_to_file(info, ofn)


def run_a2g_epoch(data_iter, model, criterion, optimizer,
                  model_name="a2g", desc="Train", curr_epoch=0,
                  logs_dir=None, max_grad_norm=5.0, report_acc=False):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct_k = 0
    total_correct_2k = 0
    total_correct_10 = 0
    total_correct_20 = 0
    total_correct_50 = 0
    total_correct_100 = 0
    total_acc_tokens = 0
    for batch in tqdm(data_iter, mininterval=2, desc=desc, leave=False, ascii=True):
        g_log_wid_probs = model.forward(batch)
        gen_targets = batch[DK_TGT_GEN_WID].to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        g_log_wid_probs = g_log_wid_probs.view(-1, g_log_wid_probs.size(-1))
        loss = criterion(g_log_wid_probs, gen_targets.contiguous().view(-1))
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
            n_correct_k = 0
            n_correct_2k = 0
            n_correct_10 = 0
            n_correct_20 = 0
            n_correct_50 = 0
            n_correct_100 = 0
            for bi in range(batch[DK_BATCH_SIZE]):
                k = batch[DK_WI_N_WORDS][bi]
                prob = g_log_wid_probs[bi,:].squeeze()
                _, top_100_ids = prob.topk(100)
                pred_100_ids = top_100_ids.tolist()
                pred_k_ids = set(pred_100_ids[:k])
                pred_2k_ids = set(pred_100_ids[:int(2*k)])
                pred_10_ids = set(pred_100_ids[:10])
                pred_20_ids = set(pred_100_ids[:20])
                pred_50_ids = set(pred_100_ids[:50])
                pred_100_ids = set(pred_100_ids)
                truth_id = gen_targets[bi,:].squeeze().item()
                if truth_id in pred_k_ids:
                    n_correct_k += 1
                if truth_id in pred_2k_ids:
                    n_correct_2k += 1
                if truth_id in pred_10_ids:
                    n_correct_10 += 1
                if truth_id in pred_20_ids:
                    n_correct_20 += 1
                if truth_id in pred_50_ids:
                    n_correct_50 += 1
                if truth_id in pred_100_ids:
                    n_correct_100 += 1
                total_acc_tokens += 1

            total_correct_k += n_correct_k
            total_correct_2k += n_correct_2k
            total_correct_10 += n_correct_10
            total_correct_20 += n_correct_20
            total_correct_50 += n_correct_50
            total_correct_100 += n_correct_100

    elapsed = time.time() - start
    if report_acc:
        info = desc + " epoch %d loss %f top_k acc %f top_2k acc %f top_10 acc %f top_20 acc %f top_50 acc %f top_100 acc %f ppl %f elapsed time %f" % (
                curr_epoch, total_loss / total_tokens,
                total_correct_k / total_acc_tokens,
                total_correct_2k / total_acc_tokens,
                total_correct_10 / total_acc_tokens,
                total_correct_20 / total_acc_tokens,
                total_correct_50 / total_acc_tokens,
                total_correct_100 / total_acc_tokens,
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
    rv_perf = total_correct_50 / total_acc_tokens if total_acc_tokens > 0 else 0
    return rv_loss, rv_perf


def train_a2g(params, model, train_loader, criterion, optimizer,
              completed_epochs=0, eval_loader=None, best_eval_result=0,
              best_eval_epoch=0, past_eval_results=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_a2g_epoch(train_loader, model, criterion, optimizer,
                                      model_name=params.model_name, report_acc=False,
                                      max_grad_norm=params.max_gradient_norm,
                                      curr_epoch=report_epoch, logs_dir=params.logs_dir)
        if params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
            optimizer.update_learning_rate(train_loss, "max")
        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0:
                    eval_loss, eval_score = run_a2g_epoch(eval_loader, model, criterion, None,
                                                          model_name=params.model_name, report_acc=True,
                                                          curr_epoch=report_epoch, logs_dir=None, desc="Eval")
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
        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        model_checkpoint(fn, report_epoch, model, optimizer, params,
                         past_eval_results, best_eval_result, best_eval_epoch)
        print("")


# a2keg
def make_a2keg_model(src_w2v_mat, params, src_vocab_size, src_oov_vocab_size, grid):
    from archive.model_gen import RNNKEGenerator
    d_model = params.a2g_hidden_size
    n_heads = params.a2g_num_attn_heads
    dropout_prob = params.a2g_dropout_prob
    ff = PositionwiseFeedForward(d_model, params.a2g_feedforward_hidden_size)
    attn = MultiHeadedAttention(n_heads, d_model)
    pos_encode = PositionalEncoding(d_model)
    # conv = DepthwiseSeparableConv(d_model, d_model)
    # norm_ff = NormPosFeedForward(d_model, ff, dropout_prob)
    # norm_attn = NormSelfAttn(d_model, attn, dropout_prob)
    # norm_conv = NormDepthConv(d_model, conv, dropout_prob)
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PartiallyTrainableEmbedding(params.word_embedding_dim, src_w2v_mat, src_oov_vocab_size,
                                                           padding_idx=params.pad_idx)
    src_embed_layer = ResizeWrapperEmbedding(d_model, src_word_embed_layer, multiply_by_sqrt_d_model=True)
    # src_embed_layer = src_word_embed_layer
    # enc_layer = QANetEncoderBlock(norm_conv, norm_attn, norm_ff)
    enc_layer = EncoderLayer(d_model, attn, ff, dropout_prob)
    encoder = PosEncoder(pos_encode, enc_layer, params.a2g_num_encoder_layers)
    # encoder = SimpleRNN(params.word_embedding_dim, d_model, return_aggr_vector_only=True,
    #                     rnn_dir=params.a2g_rnn_dir, n_layers=params.a2g_rnn_layers)
    word_generator = RNNKEGenerator(d_model, d_model, d_model, grid.encoding_dim, grid.base_k)
    # word_generator = MultiDimKEGenerator(d_model, grid.encoding_dim, grid.base_k)
    # word_generator = ConvKEGenerator(d_model, grid.encoding_dim, grid.base_k)
    model = Attn2Gen(
        src_embed_layer,
        encoder,
        word_generator,
        dropout_prob
    )
    return model


def eval_a2keg(model, loader, i2w, criterion, grid, k=10, desc="Eval"):
    # ofn = params.logs_dir + params.model_name + "_"+desc.lower()+"_out.txt"
    total_loss = 0
    total_correct = 0
    n_insts = 0
    # write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        probs = model(batch)
        targets = batch[DK_TGT_WKE].to(device())
        probs_out = probs
        probs = probs.view(-1, grid.base_k)
        targets = targets.contiguous().view(-1)
        loss = criterion(probs, targets)
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        for bi in range(batch[DK_BATCH_SIZE]):
            prob = probs_out[bi, :].squeeze()
            top_ids = grid.find_topk_word_ids(prob, k)
            pred_ids = [i for i in top_ids if i in i2w]
            truth_id = batch[DK_TGT_WID].squeeze()[bi].item()
            pred_words = [i2w[i] for i in pred_ids]
            truth_word = i2w[truth_id]
            # write_line_to_file("truth: " + truth_word, ofn)
            # write_line_to_file("preds top {}: {}".format( k, " ".join([w for w in pred_words if w not in exclude_tokens])), ofn)
            if truth_word in pred_words:
                total_correct += 1
        n_insts += batch[DK_BATCH_SIZE]
        total_loss += loss.item() / n_tokens
    info = "eval loss {} top{} acc {}".format(total_loss/len(loader), k, total_correct/n_insts)
    # write_line_to_file(info, ofn)
    print(info)
    return total_loss/len(loader), total_correct/n_insts


def run_a2keg_epoch(data_iter, model, criterion, optimizer, grid,
                    model_name="a2keg", desc="Train", curr_epoch=0,
                    logs_dir=None, max_grad_norm=5.0, norm_loss=True,
                    pad_word_idx=0):
    from archive.model_gen import iter_batch_word_acc
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    total_batch_acc = 0
    total_batches = 0
    for batch in tqdm(data_iter, mininterval=2, desc=desc, leave=False, ascii=True):
        probs = model(batch)
        targets = batch[DK_TGT_WKE].to(device())
        probs = probs.view(-1, grid.base_k)
        targets = targets.contiguous().view(-1)
        loss = criterion(probs, targets)
        _, preds = probs.max(1)
        b_acc = iter_batch_word_acc(preds, targets, grid, pad_word_idx=pad_word_idx)
        loss_lit = loss.item()
        loss_lit = loss_lit / batch[DK_TGT_N_TOKENS].item() if norm_loss else loss_lit
        total_loss += loss_lit
        total_tokens += batch[DK_BATCH_SIZE]
        total_batch_acc += b_acc
        total_batches += 1
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optimizer.step()
    elapsed = time.time() - start
    info = desc + " epoch %d loss %f top_1 acc %f elapsed time %f" % (
            curr_epoch, total_loss / total_batches,
            total_batch_acc / total_batches,
            elapsed)
    print(info)
    if logs_dir is not None:
        write_line_to_file(info, logs_dir + model_name + "_train_info.txt")
    return total_loss / total_tokens, total_correct / total_tokens


def train_a2keg(params, model, train_loader, criterion, optimizer, i2w, grid,
                completed_epochs=0, eval_loader=None, best_eval_result=0,
                best_eval_epoch=0, past_eval_results=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_a2keg_epoch(train_loader, model, criterion, optimizer, grid,
                                        model_name=params.model_name,
                                        max_grad_norm=params.max_gradient_norm,
                                        curr_epoch=report_epoch, logs_dir=params.logs_dir)
        if params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
            optimizer.update_learning_rate(train_loss, "max")
        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0:
                    eval_loss, eval_score = eval_a2keg(model, eval_loader, i2w, criterion, grid)
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
        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        model_checkpoint(fn, report_epoch, model, optimizer, params,
                         past_eval_results, best_eval_result, best_eval_epoch)
        print("")


def train_a2g_bce(params, model, train_loader, criterion, optimizer,
                  completed_epochs=0, eval_loader=None, best_eval_result=0,
                  best_eval_epoch=0, past_eval_results=[], past_train_loss=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_a2g_bce_epoch(train_loader, model, criterion, optimizer,
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
                    eval_loss, eval_score = run_a2g_bce_epoch(eval_loader, model, criterion, None,
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


def run_a2g_bce_epoch(data_iter, model, criterion, optimizer,
                      model_name="a2g", desc="Train", curr_epoch=0, pad_idx=0,
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
        info = desc + " epoch %d loss %f top_k acc %f top_2k acc %f top_10 acc %f top_25 acc %f top_50 acc %f top_100 acc %f top_500 acc % f ppl %f elapsed time %f" % (
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
