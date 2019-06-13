import copy
import random
from constants import *
from embeddings import *
from utils.misc_utils import *
from tqdm import tqdm
from utils.model_utils import parallel, device, model_checkpoint
from utils.eval_utils import corpus_eval


class DVSeq2Seq(nn.Module):

    def __init__(self, params, src_embed, target_word_embed, encoder, decoder):
        super(DVSeq2Seq, self).__init__()
        self.params = params
        self.src_embed = src_embed
        self.target_word_embed = parallel(target_word_embed)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_ff = parallel(nn.Linear(encoder.hidden_size * encoder.rnn_dir, encoder.hidden_size * encoder.rnn_dir))

    def forward(self, data_dict):
        tgt = data_dict[DK_TGT_GEN_WID].to(device())
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        target_len = tgt.shape[1]
        encoder_op, encoder_hidden = self.encode(data_dict)
        decoder_hidden = self.prep_enc_hidden_for_dec(encoder_hidden)
        use_teacher_forcing = True if random.random() < self.params.s2s_teacher_forcing_ratio else False
        g_probs, attns, c_probs = self.decode(encoder_op, decoder_hidden, src_mask, target_len,
                                              tgt=tgt if use_teacher_forcing else None)
        return g_probs, attns, c_probs

    def encode(self, data_dict):
        src = data_dict[DK_SRC_WID].to(device())
        src_mask = data_dict[DK_SRC_WID_MASK]
        encoder_hidden = None
        encoder_cell = None
        src_lens = torch.sum(src_mask.squeeze(1), dim=1)
        if self.params.s2s_encoder_type.lower() == "lstm": encoder_hidden = (encoder_hidden, encoder_cell)
        src = self.src_embed(src)
        encoder_op, encoder_hidden = self.encoder(src, encoder_hidden, src_lens)
        return encoder_op, encoder_hidden

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, 1, self.params.s2s_encoder_hidden_size * self.params.s2s_encoder_rnn_dir)
        return context.data.new(*h_size).zero_().float().to(device())

    def prep_enc_hidden_for_dec(self, encoder_hidden):
        decoder_hidden = encoder_hidden
        if self.params.s2s_encoder_type.lower() == "lstm":
            if self.params.s2s_decoder_type.lower() == "lstm":
                if self.params.s2s_encoder_rnn_dir == 2:
                    h = torch.cat([encoder_hidden[0][0, :, :], encoder_hidden[0][1, :, :]], dim=1).unsqueeze(0)
                    c = torch.cat([encoder_hidden[1][0, :, :], encoder_hidden[1][1, :, :]], dim=1).unsqueeze(0)
                else:
                    h = encoder_hidden[0]
                    c = encoder_hidden[1]
                decoder_hidden = (torch.tanh(self.hidden_ff(h)), c)
            else:
                if self.params.s2s_encoder_rnn_dir == 2:
                    decoder_hidden = torch.cat([encoder_hidden[0][0, :, :], encoder_hidden[0][1, :, :]], dim=1).unsqueeze(0)
                else:
                    decoder_hidden = encoder_hidden[0]
                decoder_hidden = torch.tanh(self.hidden_ff(decoder_hidden))
        elif self.params.s2s_encoder_type.lower() == "gru":
            if self.params.s2s_decoder_type.lower() == "lstm":
                if self.params.s2s_encoder_rnn_dir == 2:
                    h = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
                    c = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
                else:
                    h = encoder_hidden
                    c = encoder_hidden
                decoder_hidden = (torch.tanh(self.hidden_ff(h)), c)
            else:
                if self.params.s2s_encoder_rnn_dir == 2:
                    decoder_hidden = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
                else:
                    decoder_hidden = encoder_hidden
                decoder_hidden = torch.tanh(self.hidden_ff(decoder_hidden))
        return decoder_hidden # B x E

    def decode_step(self, decoder_input, decoder_hidden, encoder_past_hs, src_mask, context, precomp):
        decoder_input = self.target_word_embed(decoder_input)
        decoder_output, decoder_attn, copy_prob, hidden, context, precomp = self.decoder(decoder_input, decoder_hidden,
                                                                                         encoder_past_hs, context, precomp,
                                                                                         batch_enc_attn_mask=src_mask)
        return decoder_output, decoder_attn, copy_prob, hidden, context, precomp

    def decode(self, enc_hs, dec_hidden, src_mask, target_length, tgt=None):
        sos = torch.ones(enc_hs.shape[0],1).fill_(self.params.sos_idx).long().to(device())
        gen_probs = []
        cpy_probs = []
        dec_attns = []
        dec_input = sos
        context = self.make_init_att(dec_hidden)
        precomp = None
        for di in range(target_length):
            dec_output, dec_attn, cpy_prob, dec_hidden, context, precomp = self.decode_step(dec_input, dec_hidden, enc_hs, src_mask, context, precomp)
            if tgt is not None:
                dec_input = tgt[:, di].unsqueeze(1)
            else:
                _, next_wi = dec_output.topk(1)
                dec_input = next_wi.squeeze(2).detach().long().to(device())
            gen_probs.append(dec_output)
            cpy_probs.append(cpy_prob)
            dec_attns.append(dec_attn)
        return torch.cat(gen_probs, dim=1), torch.cat(dec_attns, dim=1), torch.cat(cpy_probs, dim=1)


class MHACMDecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, tgt_vocab_size, n_layers=1, rnn_type="gru", max_pool_size=2, dropout_prob=0.1):
        super(MHACMDecoderRNN, self).__init__()
        self.max_pool_size = max_pool_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.concat_compression = parallel(nn.Linear(self.hidden_size * 2 + self.input_size, self.hidden_size))
        self.max_out = MaxOut(self.max_pool_size)
        self.attn = MHAttn(num_heads=8, d_model=self.hidden_size)
        self.generator = parallel(nn.Linear(int(self.hidden_size / self.max_pool_size), self.tgt_vocab_size))
        self.dropout = nn.Dropout(dropout_prob)
        self.copy_switch = parallel(nn.Linear(self.hidden_size * 2, 1))
        if self.rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size + self.hidden_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=False)
        else:
            self.rnn = nn.GRU(input_size=self.input_size + self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.n_layers,
                              dropout=self.dropout_prob if self.n_layers>1 else 0,
                              batch_first=True,
                              bidirectional=False)
        init_weights(self, base_model_type="rnn")

    def forward(self, embedding, hidden, encoder_outputs, context, precomp, batch_enc_attn_mask=None):
        word_embed = embedding
        embedding = torch.cat([embedding, context.repeat(1, embedding.shape[1], 1)], dim=2)
        rnn_output, hidden = self.rnn(embedding, hidden)
        attn_weights, context = self.attn(rnn_output, encoder_outputs, batch_enc_attn_mask)
        concat_input = torch.cat([word_embed, rnn_output, context], dim=2)
        concat_output = torch.tanh(self.concat_compression(concat_input))
        concat_output = self.max_out(concat_output)
        concat_output = self.dropout(concat_output)
        output = self.generator(concat_output)
        output = F.softmax(output, dim=2)
        # copy mechanism
        copy_prob = self.copy_switch(torch.cat((rnn_output, context), dim=2))
        copy_prob = torch.sigmoid(copy_prob)
        if torch.isnan(output).any(): assert False, "NaN detected in output when decoding with attention!"
        if torch.isnan(attn_weights).any(): assert False, "NaN detected in attn_weights when decoding with attention!"
        return output, attn_weights, copy_prob, hidden, context, precomp


class MHAttn(nn.Module):

    def __init__(self, num_heads, d_model):
        super(MHAttn, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.q_ff = nn.Linear(d_model, d_model)
        self.k_ff = nn.Linear(d_model, d_model)
        self.score_comb = nn.Linear(self.h, 1)
        init_weights(self)

    def forward(self, query, key, mask=None):
        batch_size = query.size(0)

        query = self.q_ff(query)
        key = self.k_ff(key)

        query = query.contiguous().view(batch_size, self.h, -1, self.d_k)
        key = key.contiguous().view(batch_size, self.h, -1, self.d_k)

        mh_hidden = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = mh_hidden
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        p_attn = F.softmax(attn_scores, dim=-1)
        p_attn = p_attn.contiguous()

        context = p_attn.matmul(key)
        context = context.contiguous().view(batch_size, -1, self.h * self.d_k)

        probs = self.score_comb(mh_hidden.transpose(1, 3)).transpose(1, 3).squeeze(1)
        probs = F.softmax(probs, dim=-1).contiguous()

        return probs, context


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
            # rv = hidden
            # if self.rnn_dir == 2:
            #     rv = torch.cat([hidden[0, :].unsqueeze(1), hidden[1, :].unsqueeze(1)], dim=2).transpose(0,1)
            # rv = rv.transpose(0,1)
            return rv
        elif self.return_output_vector_only:
            return outputs
        else:
            return outputs, hidden


class S2SBeamSearchResult:

    def __init__(self, enc_h, i2w, idx_in_batch, src_seg_list,
                 beam_width=4, sos_idx=2, eos_idx=3, ctx=None,
                 gamma=0.0, len_norm=0.0):
        self.idx_in_batch = idx_in_batch
        self.beam_width = beam_width
        self.gamma = gamma
        self.len_norm = len_norm
        self.eos_idx = eos_idx
        self.src_seg_list = src_seg_list
        sos = torch.ones(1,1).fill_(sos_idx).long().to(device())
        self.curr_candidates = [
            (sos, 0.0, [], [i2w[sos_idx]], enc_h, ctx)
        ]
        self.completed_insts = []
        self.done = False

    def update(self, probs, next_vals, next_wids, next_words, dec_hs, ctx=None):
        assert len(next_wids) == len(self.curr_candidates)
        next_candidates = []
        for i, tup in enumerate(self.curr_candidates):
            score = tup[1]
            prev_prob_list = [t for t in tup[2]]
            prev_words = [t for t in tup[3]]
            decoder_hidden = dec_hs[i]
            context = ctx[i] if ctx is not None else None
            preds = next_wids[i]
            vals = next_vals[i]
            pred_words = next_words[i]
            prev_prob_list.append(probs)
            for bi in range(len(preds)):
                wi = preds[bi]
                val = vals[bi]
                word = pred_words[bi]
                div_penalty = 0.0
                if i > 0: div_penalty = self.gamma * (bi+1)
                new_score = score + val - div_penalty
                new_tgt = torch.ones(1,1).long().fill_(wi).to(device())
                new_words = [w for w in prev_words]
                new_words.append(word)
                if wi == self.eos_idx:
                    if self.len_norm > 0:
                        length_penalty = (self.len_norm + new_tgt.shape[1]) / (self.len_norm + 1)
                        new_score /= length_penalty ** self.len_norm
                    else:
                        new_score = new_score / new_tgt.shape[1] if new_tgt.shape[1] > 0 else new_score
                    ppl = 0 # TODO: add perplexity later
                    self.completed_insts.append((new_tgt, new_score, ppl, new_words))
                else:
                    next_candidates.append((new_tgt, new_score, prev_prob_list, new_words, decoder_hidden, context))
        next_candidates = sorted(next_candidates, key=lambda t: t[1], reverse=True)
        next_candidates = next_candidates[:self.beam_width]
        self.curr_candidates = next_candidates
        self.done = len(self.curr_candidates) == 0

    def get_curr_tgt(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[0] for tup in self.curr_candidates], dim=0).long().to(device())

    def get_curr_dec_hidden(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[4] for tup in self.curr_candidates], dim=1).float().to(device())

    def get_curr_context(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[5] for tup in self.curr_candidates if tup[5] is not None], dim=0).float().to(device())

    def get_curr_candidate_size(self):
        return len(self.curr_candidates)

    def get_curr_src_seg_list(self):
        return [self.src_seg_list for _ in range(len(self.curr_candidates))]

    def collect_results(self, topk=1):
        for cand in self.curr_candidates:
            self.completed_insts.append((cand[0], cand[1], 0, cand[3])) # TODO: for now perplexity is 0
        self.completed_insts = sorted(self.completed_insts, key=lambda t: t[1], reverse=True)
        return self.completed_insts[:topk]


def get_gen_cpy_log_probs(g_wid_probs, c_wid_probs, c_gate_probs):
    c_wid_probs = c_wid_probs * c_gate_probs.expand_as(c_wid_probs) + 1e-8
    g_wid_probs = g_wid_probs * (1 - c_gate_probs).expand_as(g_wid_probs) + 1e-8
    c_log_wid_probs = torch.log(c_wid_probs)
    g_log_wid_probs = torch.log(g_wid_probs)
    return g_log_wid_probs, c_log_wid_probs


def make_dv_seq2seq_model(src_w2v_mat, tgt_w2v_mat, params, src_vocab_size, tgt_vocab_size, same_word_embedding=False):
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=params.word_embed_further_training)
    if tgt_w2v_mat is None:
        tgt_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, tgt_vocab_size)
    else:
        tgt_word_embed_layer = PreTrainedWordEmbedding(tgt_w2v_mat, params.word_embedding_dim,
                                                       allow_further_training=params.word_embed_further_training)
    if same_word_embedding:
        tgt_word_embed_layer = src_word_embed_layer
    src_embed_layer = src_word_embed_layer
    encoder = SimpleRNN(params.word_embedding_dim, params.dv_seq2seq_hidden_size,
                        dropout_prob=params.dv_seq2seq_dropout_prob, rnn_dir=params.dv_seq2seq_encoder_rnn_dir)
    decoder = MHACMDecoderRNN(params.word_embedding_dim,
                              params.dv_seq2seq_hidden_size * params.dv_seq2seq_encoder_rnn_dir,
                              tgt_vocab_size,
                              dropout_prob=params.dv_seq2seq_dropout_prob)
    model = DVSeq2Seq(params, src_embed_layer, tgt_word_embed_layer, encoder, decoder)
    return model


def train_dv_seq2seq(params, model, train_loader, criterion_gen, criterion_cpy, optimizer,
                     completed_epochs=0, eval_loader=None, best_eval_result=0, best_eval_epoch=0,
                     past_eval_results=[], checkpoint=True):
    model = model.to(device())
    criterion_gen = criterion_gen.to(device())
    criterion_cpy = criterion_cpy.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_dv_seq2seq_epoch(model, train_loader, criterion_gen, criterion_cpy,
                                             curr_epoch=report_epoch, optimizer=optimizer,
                                             max_grad_norm=params.max_gradient_norm,
                                             desc="Train", pad_idx=params.pad_idx,
                                             model_name=params.model_name,
                                             logs_dir=params.logs_dir)
        if params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
            optimizer.update_learning_rate(train_loss, "max")
        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0:
                    eval_score = eval_dv_seq2seq(model, eval_loader, params)
                    if eval_score> best_eval_result:
                        best_eval_result = eval_score
                        best_eval_epoch = report_epoch
                        print("Model best checkpoint with score {}".format(eval_score))
                        fn = params.saved_models_dir + params.model_name + "_best.pt"
                        if checkpoint:
                            model_checkpoint(fn, report_epoch, model, optimizer, params,
                                             past_eval_results, best_eval_result, best_eval_epoch)
                    info = "Best {} so far {} from epoch {}".format(params.eval_metric, best_eval_result, best_eval_epoch)
                    print(info)
                    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
                    if hasattr(optimizer, "update_learning_rate") and not params.lr_decay_with_train_perf:
                        optimizer.update_learning_rate(eval_score)
                    past_eval_results.append(eval_score)
                    if len(past_eval_results) > params.past_eval_scores_considered:
                        past_eval_results = past_eval_results[1:]
        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        if checkpoint:
            model_checkpoint(fn, report_epoch, model, optimizer, params,
                             past_eval_results, best_eval_result, best_eval_epoch)
        print("")
    return best_eval_result, best_eval_epoch


def run_dv_seq2seq_epoch(model, loader, criterion_gen, criterion_cpy,
                         curr_epoch=0, max_grad_norm=5.0, optimizer=None,
                         desc="Train", pad_idx=0, model_name="dv_seq2seq",
                         logs_dir=""):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        g_wid_probs, c_wid_probs, c_gate_probs = model(batch)
        gen_targets = batch[DK_TGT_GEN_WID].to(device())
        cpy_targets = batch[DK_TGT_CPY_WID].to(device())
        cpy_truth_gates = batch[DK_TGT_CPY_GATE].to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        g_log_wid_probs, c_log_wid_probs = get_gen_cpy_log_probs(g_wid_probs, c_wid_probs, c_gate_probs)
        c_log_wid_probs = c_log_wid_probs * (cpy_truth_gates.unsqueeze(2).expand_as(c_log_wid_probs))
        g_log_wid_probs = g_log_wid_probs * ((1 - cpy_truth_gates).unsqueeze(2).expand_as(g_log_wid_probs))
        g_log_wid_probs = g_log_wid_probs.view(-1, g_log_wid_probs.size(-1))
        c_log_wid_probs = c_log_wid_probs.view(-1, c_log_wid_probs.size(-1))
        g_loss = criterion_gen(g_log_wid_probs, gen_targets.contiguous().view(-1))
        c_loss = criterion_cpy(c_log_wid_probs, cpy_targets.contiguous().view(-1))
        loss = g_loss + c_loss
        # compute acc
        tgt = copy.deepcopy(gen_targets.view(-1, 1).squeeze(1))
        c_sw = cpy_truth_gates.view(-1, 1).squeeze(1)
        c_tg = cpy_targets.view(-1, 1).squeeze(1)
        g_preds_i = copy.deepcopy(g_log_wid_probs.max(1)[1])
        c_preds_i = c_log_wid_probs.max(1)[1]
        g_preds_v = g_log_wid_probs.max(1)[0]
        for i in range(g_preds_i.shape[0]):
            if g_preds_v[i] == 0: g_preds_i[i] = c_preds_i[i]
        for i in range(tgt.shape[0]):
            if c_sw[i] == 1: tgt[i] = c_tg[i]
        n_correct = g_preds_i.data.eq(tgt.data)
        n_correct = n_correct.masked_select(tgt.ne(pad_idx).data).sum()
        total_loss += loss.item()
        total_correct += n_correct.item()
        total_tokens += n_tokens
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,model.parameters()), max_grad_norm)
            optimizer.step()
    loss_report = total_loss / total_tokens
    acc = total_correct / total_tokens
    elapsed = time.time() - start
    info = desc + " epoch %d loss %f, acc %f ppl %f elapsed time %f" % (curr_epoch, loss_report, acc,
                                                                        math.exp(loss_report), elapsed)
    print(info)
    write_line_to_file(info, logs_dir + model_name + "_train_info.txt")
    return loss_report, acc


def eval_dv_seq2seq(model, loader, params, desc="Eval"):
    start = time.time()
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    truth_rsp = []
    gen_rsp = []
    ofn = params.logs_dir + params.model_name + "_out.txt"
    write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        batch = copy.deepcopy(batch)
        beam_rvs = dv_seq2seq_beam_decode_batch(model, batch, params.sos_idx, params.tgt_i2w, eos_idx=params.eos_idx,
                                                len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                                max_len=params.max_decoded_seq_len, beam_width=params.beam_width_eval)
        for bi in range(batch[DK_BATCH_SIZE]):
            best_rv = beam_rvs[bi][0][3]
            truth_rsp.append([[w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]])
            gen_rsp.append([w for w in best_rv if w not in exclude_tokens])
            write_line_to_file("truth: " + " ".join([w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]), ofn)
            write_line_to_file("pred: " + " ".join([w for w in best_rv if w not in exclude_tokens]), ofn)
    perf = corpus_eval(gen_rsp, truth_rsp)
    elapsed = time.time() - start
    info = "Full eval result {} elapsed {}".format(str(perf), elapsed)
    print(info)
    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
    return perf[params.eval_metric]


def dv_seq2seq_beam_decode_batch(model, batch, start_idx, i2w, max_len,
                                 gamma=0.0, oov_idx=1, beam_width=4, eos_idx=3,
                                 len_norm=1.0, topk=1):
    batch_size = batch[DK_SRC_WID].shape[0]
    model = model.to(device())
    encoder_op, encoder_hidden = model.encode(batch)
    src_mask = batch[DK_SRC_WID_MASK].to(device())
    encoder_hidden = model.prep_enc_hidden_for_dec(encoder_hidden)
    context = model.make_init_att(encoder_hidden)
    batch_results = [
        S2SBeamSearchResult(idx_in_batch=bi, i2w=i2w, src_seg_list=batch[DK_SRC_SEG_LISTS][bi],
                            enc_h=encoder_hidden[:, bi, :].unsqueeze(0),
                            beam_width=beam_width, sos_idx=start_idx,
                            eos_idx=eos_idx, ctx=context[bi,:].unsqueeze(0),
                            gamma=gamma, len_norm=len_norm)
        for bi in range(batch_size)
    ]
    final_ans = []
    for i in range(max_len):
        curr_actives = [b for b in batch_results if not b.done]
        if len(curr_actives) == 0: break
        b_tgt_list = [b.get_curr_tgt() for b in curr_actives]
        b_tgt = torch.cat(b_tgt_list, dim=0)
        b_hidden_list = [b.get_curr_dec_hidden() for b in curr_actives]
        b_hidden = torch.cat(b_hidden_list, dim=1).to(device())
        b_ctx_list = [b.get_curr_context() for b in curr_actives]
        b_context = torch.cat(b_ctx_list, dim=0).to(device())
        b_cand_size_list = [b.get_curr_candidate_size() for b in curr_actives]
        b_src_seg_list = [b.get_curr_src_seg_list() for b in curr_actives]
        enc_op = torch.cat([encoder_op[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in curr_actives], dim=0)
        s_mask = torch.cat([src_mask[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in curr_actives], dim=0)
        gen_wid_probs, cpy_wid_probs, cpy_gate_probs, b_hidden, b_context, _ = model.decode_step(
            b_tgt, b_hidden, enc_op, s_mask, b_context, None)
        gen_wid_probs, cpy_wid_probs = get_gen_cpy_log_probs(gen_wid_probs, cpy_wid_probs, cpy_gate_probs)
        comb_prob = torch.cat([gen_wid_probs, cpy_wid_probs], dim=2)
        beam_i = 0
        for bi, size in enumerate(b_cand_size_list):
            g_probs = comb_prob[beam_i:beam_i+size, :].view(size, -1, comb_prob.size(-1))
            hiddens = b_hidden[:, beam_i:beam_i+size, :].view(size, -1, b_hidden.size(-1))
            ctxs = b_context[beam_i:beam_i+size, :].view(size, -1, b_context.size(-1))
            vt, it = g_probs.topk(beam_width)
            next_vals, next_wids, next_words, dec_hs, ctx = [], [], [], [], []
            for ci in range(size):
                vals, wis, words = [], [], []
                for idx in range(beam_width):
                    vals.append(vt[ci,0,idx].item())
                    wi = it[ci,0,idx].item()
                    if wi in i2w:  # generate
                        word = i2w[wi]
                    else:  # copy
                        c_wi = wi - len(i2w)
                        if c_wi < len(b_src_seg_list[bi][ci]):
                            word = b_src_seg_list[bi][ci][c_wi]
                        else:
                            word = i2w[oov_idx]
                        wi = oov_idx
                    wis.append(wi)
                    words.append(word)
                next_vals.append(vals)
                next_wids.append(wis)
                next_words.append(words)
            dec_hs = [hiddens[j,:].unsqueeze(1) for j in range(hiddens.shape[0])]
            ctx = [ctxs[j,:].unsqueeze(0) for j in range(ctxs.shape[0])]
            curr_actives[bi].update(g_probs, next_vals, next_wids, next_words, dec_hs, ctx)
            beam_i += size
    for b in batch_results:
        final_ans.append(b.collect_results(topk=topk))
    return final_ans



