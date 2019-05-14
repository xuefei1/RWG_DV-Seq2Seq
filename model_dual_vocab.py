import time
import math
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from constants import *
from tqdm import tqdm
from utils.eval_utils import corpus_eval
from utils.model_utils import device, model_checkpoint, init_weights, parallel
from utils.misc_utils import write_line_to_file
from embeddings import TrainableEmbedding, PreTrainedWordEmbedding
from components import *
from data_loaders import word_idx_tsr_to_data
from utils.lang_utils import gen_word2idx_vec_rep, pad_seg_lists_in_place
from utils.misc_utils import merge_batch_word_seg_lists


def get_sv_dv_log_probs(sv_probs, dv_probs, dv_gates):
    dv_probs = dv_probs * dv_gates.expand_as(dv_probs) + 1e-8
    sv_probs = sv_probs * (1 - dv_gates).expand_as(sv_probs) + 1e-8
    dv_log_probs = torch.log(dv_probs)
    sv_log_probs = torch.log(sv_probs)
    return sv_log_probs, dv_log_probs


class CatAttention(nn.Module):

    def __init__(self, attend_dim, query_dim, att_dim):
        super(CatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        init_weights(self)

    def forward(self, input, context, mask=None):
        precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
        precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input) # batch x 1 x att_dim
        try:
            tmp10 = precompute + targetT.repeat(1,precompute.shape[1],1)  # batch x sourceL x att_dim
        except RuntimeError:
            print(input.size())
            print(context.size())
            print(targetT.size())
            print(precompute.size())
            assert False
        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if mask is not None:
            mask = (mask == 0).float().squeeze(1).to(device())
            energy = energy * (1 - mask) + mask * (-1000000)
        score = self.sm(energy)
        score_m = score.view(score.size(0), 1, score.size(1))  # batch x 1 x sourceL
        # weightedContext = torch.bmm(score_m, context)  # batch x 1 x dim
        return score_m


class MHParallelAttention(nn.Module):

    def __init__(self, num_heads, d_model):
        super(MHParallelAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        # self.d_k = d_model
        self.h = num_heads
        # self.q_ff = nn.Linear(d_model, d_model)
        # self.k_ff = nn.Linear(d_model, d_model)
        self.q_ff = nn.Linear(self.d_k, self.d_k)
        self.k_ff = nn.Linear(self.d_k, self.d_k)
        self.score_comb = nn.Linear(self.h, 1)
        init_weights(self)

    def forward(self, query, key, mask=None):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query = query.contiguous().view(batch_size, self.h, -1, self.d_k)
        key = key.contiguous().view(batch_size, self.h, -1, self.d_k)

        query = torch.tanh(self.q_ff(query))
        key = torch.tanh(self.k_ff(key))

        scores = torch.matmul(query, key.transpose(-2, -1))
        # attn_rv = []
        # for i in range(query.shape[2]):
        #     q = query[:,:,i,:].unsqueeze(2)
        #     q = self.q_ff(q)
        #     q = q.repeat(1,1,key.shape[2],1)
        #     q_attn_key = q + key
        #     q_attn_key = torch.tanh(q_attn_key)
        #     q_attn_key = q_attn_key.unsqueeze(2)
        #     attn_rv.append(q_attn_key)
        # attn_rv = torch.cat(attn_rv, dim=2)
        # attn_rv = self.attn_aggr(attn_rv).squeeze(4)
        # attn_rv = attn_rv.contiguous().view(batch_size, -1, key.shape[2], self.h)
        # scores = self.head_aggr(attn_rv).squeeze(3)

        scores = self.score_comb(scores.transpose(1, 3)).transpose(1, 3)
        scores = scores.squeeze(1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = p_attn.contiguous()
        return p_attn


def merge_into_2d_mask(row_mask, col_mask):
    row_mask = row_mask.squeeze()
    col_mask = col_mask.squeeze()
    assert len(row_mask.shape) == 2 and len(col_mask.shape) == 2
    assert row_mask.shape[0] == col_mask.shape[0], "batch size must equal"
    row_size, col_size = row_mask.shape[1], col_mask.shape[1]
    row_mask_repeated = row_mask.unsqueeze(2).repeat(1, 1, col_size)
    col_mask_repeated = col_mask.unsqueeze(1).repeat(1, row_size, 1)
    assert row_mask_repeated.size() == col_mask_repeated.size()
    rv_mask = row_mask_repeated & col_mask_repeated
    return rv_mask


class Seq2AttnDVSelector(nn.Module):

    def __init__(self, src_embed_layer, rnn, pos_encoding_layer, attn_layer, word_generator,
                 n_attn_layers=1, dropout_prob=0.0):
        super(Seq2AttnDVSelector, self).__init__()
        self.src_embed_layer = parallel(src_embed_layer)
        self.rnn = parallel(rnn)
        self.pos_encoding_layer = pos_encoding_layer
        self.attn_layers = clones(parallel(attn_layer), n_attn_layers)
        self.word_generator = parallel(word_generator)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, data_dict):
        src = data_dict[DK_SRC_WID].to(device())
        src = self.src_embed_layer(src)
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        rnn_op = self.rnn(src, mask=src_mask)
        q = rnn_op[:, -1, :].unsqueeze(1)
        k = self.pos_encoding_layer(rnn_op)
        v = k
        output = q
        for layer in self.attn_layers:
            output = layer(q, k, v, src_mask)
        assert output.shape[1] == 1
        output = self.dropout(output)
        word_probs = self.word_generator(output)
        return word_probs


class SelectorWrapper(nn.Module):

    def __init__(self, selector, embedding_layer, vocab,
                 pad_idx=0, exclude_word_indices=set(), topk=100):
        super(SelectorWrapper, self).__init__()
        self.selector = selector
        self.embedding_layer = copy.deepcopy(embedding_layer)
        self.topk = topk
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.exclude_word_indices = exclude_word_indices

    def forward(self, data_dict, new_topk=None):
        topk = new_topk if new_topk is not None else self.topk
        dv_word_probs = self.selector(data_dict)
        _, dv_word_indices = dv_word_probs.topk(dim=-1, k=topk)
        words = word_idx_tsr_to_data(dv_word_indices.squeeze(1), self.vocab.i2w,
                                     self.pad_idx, self.exclude_word_indices,
                                     word_delim=None)
        # manually add input words
        src_seg_lists = data_dict[DK_SRC_SEG_LISTS]
        rv_words = merge_batch_word_seg_lists(src_seg_lists, words, remove_duplicate_words=True)
        dv_word_indices = gen_word2idx_vec_rep(rv_words, self.vocab.w2i, max([len(l) for l in rv_words]),
                                               pad_idx=self.vocab.pad_idx, oov_idx=self.vocab.oov_idx)
        dv_word_indices = torch.from_numpy(dv_word_indices).long().to(device())
        rv_mask = dv_word_indices.ne(self.pad_idx).byte().to(device())
        return rv_words, rv_mask


class SDWrapper(nn.Module):

    def __init__(self, selector, discriminator, embedding_layer, selector_tgt_vocab,
                 discriminator_cdw_vocab, discriminator_label_vocab,
                 pad_idx=0, exclude_word_indices=set(), topk=100, fallback_topk=10):
        super(SDWrapper, self).__init__()
        self.selector = selector
        self.discriminator = discriminator
        self.embedding_layer = copy.deepcopy(embedding_layer)
        self.topk = topk
        self.selector_tgt_vocab = selector_tgt_vocab
        self.discriminator_cdw_vocab = discriminator_cdw_vocab
        self.discriminator_label_vocab = discriminator_label_vocab
        self.pad_idx = pad_idx
        self.fallback_topk = fallback_topk
        self.exclude_word_indices = exclude_word_indices

    def forward(self, data_dict, new_topk=None):
        topk = new_topk if new_topk is not None else self.topk
        dv_word_probs = self.selector(data_dict)
        _, dv_word_indices = dv_word_probs.topk(dim=-1, k=topk)
        word_seg_lists = word_idx_tsr_to_data(dv_word_indices.squeeze(1), self.selector_tgt_vocab.i2w,
                                              self.pad_idx, self.exclude_word_indices,
                                              word_delim=None)

        cdw_tsr_seq_len = max([len(l) for l in word_seg_lists])
        cdw_vec = gen_word2idx_vec_rep(word_seg_lists, self.discriminator_cdw_vocab.w2i, cdw_tsr_seq_len,
                                       pad_idx=self.discriminator_cdw_vocab.pad_idx,
                                       oov_idx=self.discriminator_cdw_vocab.oov_idx)
        cdw_oov_vec = gen_word2idx_vec_rep(word_seg_lists, self.discriminator_cdw_vocab.oov_w2i, cdw_tsr_seq_len,
                                           pad_idx=self.discriminator_cdw_vocab.pad_idx,
                                           oov_idx=self.discriminator_cdw_vocab.pad_idx)
        cdw = torch.from_numpy(cdw_vec).long()
        cdw_oov = torch.from_numpy(cdw_oov_vec).long()
        data = copy.deepcopy(data_dict)
        data[DK_CAND_WORD_WID] = cdw
        data[DK_CAND_WORD_OOV_WID] = cdw_oov
        data[DK_CAND_WORD_SEG_LISTS] = word_seg_lists

        dv_word_label_probs = self.discriminator(data)
        dv_word_labels = dv_word_label_probs.squeeze().max(2)[1]
        dv_word_labels = dv_word_labels.tolist()
        rv_words = []
        dv_word_indices = dv_word_indices.squeeze()
        for bi, pred_words in enumerate(word_seg_lists):
            tmp_words = []
            for i, word in enumerate(pred_words):
                word_idx = dv_word_indices[bi, i].item()
                label = dv_word_labels[bi][i]
                text_label = self.discriminator_label_vocab.i2w[label]
                if text_label == "1" and word_idx not in self.exclude_word_indices:
                    tmp_words.append(word)
            if len(tmp_words) == 0:
                tmp_words = pred_words[:self.fallback_topk]
            rv_words.append(tmp_words)

        src_seg_lists = data_dict[DK_SRC_SEG_LISTS]
        rv_words = merge_batch_word_seg_lists(src_seg_lists, rv_words, remove_duplicate_words=True)
        rv_word_indices = gen_word2idx_vec_rep(rv_words, self.selector_tgt_vocab.w2i, max([len(l) for l in rv_words]),
                                               pad_idx=self.selector_tgt_vocab.pad_idx,
                                               oov_idx=self.selector_tgt_vocab.oov_idx)
        rv_word_indices = torch.from_numpy(rv_word_indices).long().to(device())
        rv_mask = rv_word_indices.ne(self.pad_idx).byte().to(device())

        return rv_words, rv_mask


class RNNSelfAttn(nn.Module):

    def __init__(self, d_model):
        super(RNNSelfAttn, self).__init__()
        self.d_model = d_model
        self.ff = nn.Linear(d_model, d_model)
        self.attn = Attn(self.d_model, multiply_outputs=True)
        init_weights(self, base_model_type="rnn")

    def forward(self, rnn_op, mask=None):
        x = self.ff(rnn_op[:, -1, :].unsqueeze(1))
        rv = self.attn(x, rnn_op, mask)
        return rv


class Seq2AMDVSelector(nn.Module):

    def __init__(self, src_embed_layer, rnn, rnn_self_attn, ff_max_out, word_generator, dropout_prob=0.0):
        super(Seq2AMDVSelector, self).__init__()
        self.src_embed_layer = parallel(src_embed_layer)
        self.rnn = parallel(rnn)
        self.rnn_self_attn = parallel(rnn_self_attn)
        self.ff_max_out = parallel(ff_max_out)
        self.word_generator = parallel(word_generator)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, data_dict):
        src = data_dict[DK_SRC_WID].to(device())
        src = self.src_embed_layer(src)
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        rnn_op = self.rnn(src, mask=src_mask)
        context = self.rnn_self_attn(rnn_op)
        concat_input = torch.cat([rnn_op[:,-1,:].unsqueeze(1), context], dim=2)
        concat_input = self.dropout(concat_input)
        output = self.ff_max_out(concat_input)
        assert output.shape[1] == 1
        word_probs = self.word_generator(output)
        return word_probs


class AttnMaxOutDecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size,
                 n_layers=1, rnn_type="gru", max_pool_size=2, dropout_prob=0.1):
        super(AttnMaxOutDecoderRNN, self).__init__()
        self.max_pool_size = max_pool_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.concat_compression = parallel(nn.Linear(self.hidden_size * 2 + self.input_size, self.hidden_size))
        self.max_out = MaxOut(self.max_pool_size)
        self.attn = CatAttention(self.hidden_size, self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
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

    def forward(self, embedding, hidden, encoder_outputs, context, batch_enc_attn_mask=None):
        word_embed = embedding
        embedding = torch.cat([embedding, context.repeat(1, embedding.shape[1], 1)], dim=2)
        rnn_output, hidden = self.rnn(embedding, hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs, mask=batch_enc_attn_mask)
        context = torch.bmm(attn_weights, encoder_outputs)
        concat_input = torch.cat([word_embed, rnn_output, context], dim=2)
        concat_output = torch.tanh(self.concat_compression(concat_input))
        output = self.max_out(concat_output)
        if torch.isnan(output).any(): assert False, "NaN detected in output when decoding with attention!"
        if torch.isnan(attn_weights).any(): assert False, "NaN detected in attn_weights when decoding with attention!"
        return output, hidden, context


class AttnDVGenerator(nn.Module):

    def __init__(self, d_model, num_heads):
        super(AttnDVGenerator, self).__init__()
        self.d_model = d_model
        self.src2dv_attn = MHParallelAttention(num_heads, self.d_model)
        self.context_ff = nn.Linear(self.d_model * 2, self.d_model)
        self.dv_gates_ff = nn.Linear(self.d_model, 1)

    def forward(self, word_embedded, hidden, word_indices_mask):
        dv_probs = self.src2dv_attn(hidden, word_embedded, word_indices_mask)
        weighted_context = torch.bmm(dv_probs, word_embedded)
        weighted_context = self.context_ff(torch.cat([hidden, weighted_context], dim=2))
        dv_gates_hidden = self.dv_gates_ff(weighted_context)
        dv_gates = torch.sigmoid(dv_gates_hidden)
        return dv_probs, dv_gates, weighted_context


class Seq2SeqDV(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder,
                 dv_word_embed,
                 sv_generator,
                 dv_generator,
                 dv_vocab,
                 dropout_prob=0.0, sos_idx=2):
        super(Seq2SeqDV, self).__init__()
        self.src_embed = parallel(src_embed)
        self.tgt_embed = parallel(tgt_embed)
        self.dv_word_embed = parallel(dv_word_embed)
        self.encoder = encoder
        self.decoder = decoder
        self.sv_generator = parallel(sv_generator)
        self.dv_generator = parallel(dv_generator)
        self.dropout = nn.Dropout(dropout_prob)
        self.sos_idx = sos_idx
        self.dv_vocab = dv_vocab

    def forward(self, data_dict):
        tgt = data_dict[DK_TGT_WID].to(device())
        target_len = tgt.shape[1]
        encoder_op, encoder_hidden = self.encode(data_dict)
        decoder_hidden = self.prep_encoder_hidden_for_decoder(encoder_hidden)
        # enc_mask = data_dict[DK_SRC_WID_MASK].to(device())
        tgt = data_dict[DK_TGT_INPUT_WID].to(device())
        # sv_probs, dec_hiddens = self.decode(decoder_hidden, encoder_op.shape[0], target_len,
        #                                     encoder_op, enc_mask,
        #                                     tgt)

        dv_word_indices = data_dict[DK_TGT_WI_WID].to(device())
        dv_word_mask = data_dict[DK_TGT_WI_WID_MASK].to(device())
        dv_word_embedded = self.dv_word_embed(dv_word_indices)

        sv_probs, dv_probs, dv_gates = self.decode(decoder_hidden, dv_word_embedded, dv_word_mask, target_len, tgt)
        return sv_probs, dv_probs, dv_gates

    def encode(self, data_dict):
        src_wid = data_dict[DK_SRC_WID].to(device())
        src_oov_wid = data_dict[DK_SRC_OOV_WID].to(device())
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        src_lens = torch.sum(src_mask.squeeze(1), dim=1)
        src = self.src_embed(src_wid, src_oov_wid)
        encoder_op, encoder_hidden = self.encoder(src, lens=src_lens)
        return encoder_op, encoder_hidden

    @staticmethod
    def prep_encoder_hidden_for_decoder(encoder_hidden):
        if encoder_hidden.shape[0] == 2:
            decoder_hidden = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
        else:
            decoder_hidden = encoder_hidden
        return decoder_hidden # B x E

    def decode_step(self, decoder_input, decoder_hidden, dv_word_embedded, dv_word_indices_mask):
        decoder_input = self.tgt_embed(decoder_input)
        decoder_output, hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.dropout(decoder_output)
        dv_probs, dv_gates, weight_context = self.dv_generator(dv_word_embedded, decoder_output, dv_word_indices_mask)
        sv_probs = self.sv_generator(weight_context)
        return sv_probs, dv_probs, dv_gates, hidden

    # def decode_step(self, decoder_input, decoder_hidden, encoder_outputs, ctx, enc_mask):
    #     decoder_input = self.tgt_embed(decoder_input)
    #     decoder_output, hidden, ctx = self.decoder(decoder_input, decoder_hidden, encoder_outputs, ctx, enc_mask)
    #     decoder_output = self.dropout(decoder_output)
    #     probs = self.sv_generator(decoder_output)
    #     return probs, hidden, ctx

    def get_batch_tgt_wid(self, tgt_words_seg_lists, tgt_word_idx):
        dec_input = torch.zeros(len(tgt_words_seg_lists)).long().to(device())
        for bi, wl in enumerate(tgt_words_seg_lists):
            word = wl[tgt_word_idx] if tgt_word_idx < len(wl) else self.tgt_w2i[PAD_TOKEN]
            wi = self.tgt_w2i[word] if word in self.tgt_w2i else self.tgt_w2i[OOV_TOKEN]
            dec_input[bi] = wi
        return dec_input.unsqueeze(1)

    def decode(self, dec_hidden, dv_word_embedded, dv_word_mask, target_length, tgt):
        batch_size = dv_word_embedded.shape[0]
        sos = torch.ones(batch_size,1).fill_(self.sos_idx).long().to(device())
        sv_probs_list = []
        dv_probs_list = []
        dv_gates_list = []
        dec_input = sos
        decoder_hiddens = []
        for di in range(target_length):
            sv_probs, dv_probs, dv_gates, dec_hidden = self.decode_step(dec_input, dec_hidden, dv_word_embedded, dv_word_mask)
            dec_input = tgt[:, di].unsqueeze(1)
            sv_probs_list.append(sv_probs)
            dv_probs_list.append(dv_probs)
            dv_gates_list.append(dv_gates)
            decoder_hiddens.append(dec_hidden)
        return torch.cat(sv_probs_list, dim=1), torch.cat(dv_probs_list, dim=1), torch.cat(dv_gates_list, dim=1)

    # def decode(self, dec_hidden, batch_size, target_length, enc_outputs, enc_mask, tgt):
    #     sos = torch.ones(batch_size,1).fill_(self.sos_idx).long().to(device())
    #     gen_probs = []
    #     dec_input = sos
    #     decoder_hiddens = []
    #     context = torch.zeros(batch_size, 1, dec_hidden.shape[2]).float().to(device())
    #     for di in range(target_length):
    #         probs, dec_hidden, context = self.decode_step(dec_input, dec_hidden, enc_outputs, context, enc_mask)
    #         # dec_input = self.get_batch_tgt_wid(tgt_words_seg_lists, di)
    #         dec_input = tgt[:, di].unsqueeze(1)
    #         gen_probs.append(probs)
    #         decoder_hiddens.append(dec_hidden)
    #     return torch.cat(gen_probs, dim=1), torch.cat(decoder_hiddens, dim=0).transpose(0,1)


def make_s2am_dv_sel_model(src_w2v_mat, params, src_vocab_size, tgt_vocab_size):
    d_model_rnn = params.s2am_dv_sel_hidden_size
    d_model_attn = d_model_rnn * params.s2am_dv_sel_rnn_dir
    dropout_prob = params.s2am_dv_sel_dropout_prob
    pool_size = params.s2am_dv_pool_size
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim, allow_further_training=True)
        # src_word_embed_layer = PartiallyTrainableEmbedding(params.word_embedding_dim, src_w2v_mat, src_oov_vocab_size,
        #                                                    padding_idx=params.pad_idx)
    src_embed_layer = src_word_embed_layer
    rnn = SimpleRNN(params.word_embedding_dim, d_model_rnn, return_output_vector_only=True,
                    rnn_dir=params.s2am_dv_sel_rnn_dir, n_layers=params.s2am_dv_sel_rnn_layers)
    rnn_self_attn = RNNSelfAttn(d_model_attn)
    ff_max_out = FFMaxOutLayer(2 * d_model_attn, d_model_attn, pool_size=pool_size)
    word_generator = Generator(d_model_attn // pool_size, tgt_vocab_size)
    model = Seq2AMDVSelector(
        src_embed_layer,
        rnn,
        rnn_self_attn,
        ff_max_out,
        word_generator,
        dropout_prob=dropout_prob
    )
    return model


# s2a_dv_sel
def make_s2a_dv_sel_model(src_w2v_mat, params, src_vocab_size, tgt_vocab_size):
    d_model_rnn = params.s2a_dv_sel_hidden_size
    d_model_attn = d_model_rnn * params.s2a_dv_sel_rnn_dir
    dropout_prob = params.s2a_dv_sel_dropout_prob
    n_heads = params.s2a_dv_sel_num_attn_heads
    ff = PositionwiseFeedForward(d_model_attn, params.s2a_dv_sel_feedforward_hidden_size, dropout=0.0)
    attn = MultiHeadedAttention(n_heads, d_model_attn)
    pos_encoding_layer = PositionalEncoding(d_model_attn)
    if src_w2v_mat is None:
        src_word_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size)
    else:
        src_word_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim, allow_further_training=True)
        # src_word_embed_layer = PartiallyTrainableEmbedding(params.word_embedding_dim, src_w2v_mat, src_oov_vocab_size,
        #                                                    padding_idx=params.pad_idx)
    src_embed_layer = src_word_embed_layer
    rnn = SimpleRNN(params.word_embedding_dim, d_model_rnn, return_output_vector_only=True,
                    rnn_dir=params.s2a_dv_sel_rnn_dir, n_layers=params.s2a_dv_sel_rnn_layers)
    attn_layer = NormAttnFF(d_model_attn, attn, ff)
    word_generator = Generator(d_model_attn, tgt_vocab_size)
    model = Seq2AttnDVSelector(
        src_embed_layer,
        rnn,
        pos_encoding_layer,
        attn_layer,
        word_generator,
        n_attn_layers=params.s2a_dv_sel_num_attn_layers,
        dropout_prob=dropout_prob
    )
    return model


def eval_s2a_dv_sel(model, loader, params, i2w, criterion, k=10, desc="Eval"):
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


def run_s2a_dv_sel_epoch(data_iter, model, criterion, optimizer,
                         model_name="s2a_dv_sel", desc="Train", curr_epoch=0,
                         logs_dir=None, max_grad_norm=5.0):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
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
        # compute top_k acc
        n_correct = 0
        n_correct_10 = 0
        n_correct_20 = 0
        n_correct_50 = 0
        n_correct_100 = 0
        for bi in range(batch[DK_BATCH_SIZE]):
            k = batch[DK_WI_N_WORDS][bi]
            prob = g_log_wid_probs[bi,:].squeeze()
            _, top_100_ids = prob.topk(100)
            pred_100_ids = top_100_ids.tolist()
            pred_ids = pred_100_ids[:k]
            pred_10_ids = pred_100_ids[:10]
            pred_20_ids = pred_100_ids[:20]
            pred_50_ids = pred_100_ids[:50]
            truth_id = gen_targets[bi,:].squeeze().item()
            if truth_id in pred_ids:
                n_correct += 1
            if truth_id in pred_10_ids:
                n_correct_10 += 1
            if truth_id in pred_20_ids:
                n_correct_20 += 1
            if truth_id in pred_50_ids:
                n_correct_50 += 1
            if truth_id in pred_100_ids:
                n_correct_100 += 1
            total_acc_tokens += 1
            break # saves time
        total_loss += loss.item()
        total_correct += n_correct
        total_correct_10 += n_correct_10
        total_correct_20 += n_correct_20
        total_correct_50 += n_correct_50
        total_correct_100 += n_correct_100
        total_tokens += n_tokens
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optimizer.step()
            if torch.isnan(loss).any():
                assert False, "nan detected after step()"
    elapsed = time.time() - start
    info = desc + " epoch %d loss %f top_k acc %f top_10 acc %f top_20 acc %f top_50 acc %f top_100 acc %f ppl %f elapsed time %f" % (
            curr_epoch, total_loss / total_tokens,
            total_correct / total_acc_tokens,
            total_correct_10 / total_acc_tokens,
            total_correct_20 / total_acc_tokens,
            total_correct_50 / total_acc_tokens,
            total_correct_100 / total_acc_tokens,
            math.exp(total_loss / total_tokens),
            elapsed)
    print(info)
    if logs_dir is not None:
        write_line_to_file(info, logs_dir + model_name + "_train_info.txt")
    return total_loss / total_tokens, total_correct / total_acc_tokens


def train_s2a_dv_sel(params, model, train_loader, criterion, optimizer,
                     completed_epochs=0, eval_loader=None, best_eval_result=0,
                     best_eval_epoch=0, past_eval_results=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_s2a_dv_sel_epoch(train_loader, model, criterion, optimizer,
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
                    eval_loss, eval_score = run_s2a_dv_sel_epoch(eval_loader, model, criterion, None,
                                                                 model_name=params.model_name,
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


class S2SDVBeamSearchResult:

    def __init__(self, enc_h, idx_in_batch, src_seg_list,
                 beam_width=4, sos_idx=2, sos_token=SOS_TOKEN, eos_idx=3, ctx=None,
                 gamma=0.0, len_norm=0.0):
        self.idx_in_batch = idx_in_batch
        self.beam_width = beam_width
        self.gamma = gamma
        self.len_norm = len_norm
        self.eos_idx = eos_idx
        self.src_seg_list = src_seg_list
        sos = torch.ones(1,1).fill_(sos_idx).long().to(device())
        self.curr_candidates = [
            (sos, 0.0, [], [sos_token], enc_h, ctx)
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


def s2s_dv_beam_decode_batch(model, batch, start_idx, uniform_vocab, sv_i2w, dv_output_i2w, max_len,
                             gamma=0.0, oov_idx=1, beam_width=4, eos_idx=3,
                             len_norm=1.0, topk=1):
    batch_size = batch[DK_SRC_WID].shape[0]
    encoder_op, encoder_hidden = model.encode(batch)
    # src_mask = batch[DK_SRC_WID_MASK].to(device())
    encoder_hidden = model.prep_encoder_hidden_for_decoder(encoder_hidden)
    dv_word_indices = batch[DK_TGT_WI_WID].to(device())
    dv_word_mask = batch[DK_TGT_WI_WID_MASK].to(device())

    batch_results = [
        S2SDVBeamSearchResult(idx_in_batch=bi, src_seg_list=batch[DK_SRC_SEG_LISTS][bi],
                              enc_h=encoder_hidden[:, bi, :].unsqueeze(0),
                              beam_width=beam_width, sos_idx=start_idx,
                              eos_idx=eos_idx, ctx=None,
                              gamma=gamma, len_norm=len_norm)
        for bi in range(batch_size)
    ]
    final_ans = []
    for i in range(max_len):
        curr_actives = [b for b in batch_results if not b.done]
        if len(curr_actives) == 0: break
        b_tgt = torch.cat([b.get_curr_tgt() for b in curr_actives], dim=0).to(device())
        b_hidden = torch.cat([b.get_curr_dec_hidden() for b in curr_actives], dim=1).to(device())
        b_cand_size_list = [b.get_curr_candidate_size() for b in curr_actives]

        b_dv_word_indices = torch.cat([dv_word_indices[b.idx_in_batch, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1)
                                       for b in curr_actives], dim=0)
        b_dv_mask = torch.cat([dv_word_mask[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1)
                               for b in curr_actives], dim=0)
        # enc_op = torch.cat(
        #     [encoder_op[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in
        #      curr_actives], dim=0)
        # s_mask = torch.cat(
        #     [src_mask[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in
        #      curr_actives], dim=0)

        dv_word_embedded = model.dv_word_embed(b_dv_word_indices)
        sv_probs, dv_probs, dv_gates, b_hidden = model.decode_step(b_tgt, b_hidden, dv_word_embedded, b_dv_mask)

        gen_wid_probs, cpy_wid_probs = get_sv_dv_log_probs(sv_probs, dv_probs, dv_gates)
        comb_prob = torch.cat([gen_wid_probs, cpy_wid_probs], dim=2)
        beam_i = 0
        for bi, size in enumerate(b_cand_size_list):
            g_probs = comb_prob[beam_i:beam_i+size, :].view(size, -1, comb_prob.size(-1))
            hiddens = b_hidden[:, beam_i:beam_i+size, :].view(size, -1, b_hidden.size(-1))
            vt, it = g_probs.topk(beam_width)
            next_vals, next_wids, next_words, dec_hs = [], [], [], []
            for ci in range(size):
                vals, wis, words = [], [], []
                for idx in range(beam_width):
                    vals.append(vt[ci,0,idx].item())
                    wi = it[ci,0,idx].item()
                    if wi in sv_i2w:  # generate from sv
                        word = sv_i2w[wi]
                    else:  # copy from dv
                        c_wi = wi - len(sv_i2w)
                        pred_dv_word_indices = b_dv_word_indices[bi, :].tolist()
                        if c_wi < len(pred_dv_word_indices):
                            wi = pred_dv_word_indices[c_wi]
                            word = dv_output_i2w[wi]
                        else:
                            word = sv_i2w[oov_idx]
                    wi = uniform_vocab.w2i[word] if word in uniform_vocab.w2i else uniform_vocab.w2i[OOV_TOKEN]
                    wis.append(wi)
                    words.append(word)
                next_vals.append(vals)
                next_wids.append(wis)
                next_words.append(words)
            dec_hs = [hiddens[j,:].unsqueeze(1) for j in range(hiddens.shape[0])]
            curr_actives[bi].update(g_probs, next_vals, next_wids, next_words, dec_hs, ctx=None)
            beam_i += size
    for b in batch_results:
        final_ans.append(b.collect_results(topk=topk))
    return final_ans


def eval_s2s_dv(model, loader, params, desc="Eval", output_to_file=False):
    start = time.time()
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    truth_rsp = []
    gen_rsp = []
    ofn = params.logs_dir + params.model_name + "_"+desc.lower()+"_out.txt"
    if output_to_file: write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        batch = copy.deepcopy(batch)
        beam_rvs = s2s_dv_beam_decode_batch(model, batch, params.sos_idx,
                                            params.uniform_vocab,
                                            params.tgt_sv_i2w,
                                            params.dv_src_vocab.i2w,
                                            eos_idx=params.eos_idx, len_norm=params.bs_len_norm,
                                            gamma=params.bs_div_gamma, max_len=params.max_decoded_seq_len,
                                            beam_width=params.beam_width_eval)
        for bi in range(batch[DK_BATCH_SIZE]):
            best_rv = beam_rvs[bi][0][3]
            truth_rsp.append([[w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]])
            gen_rsp.append([w for w in best_rv if w not in exclude_tokens])
            if output_to_file:
                write_line_to_file("truth: " + " ".join([w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]), ofn)
                write_line_to_file("pred: " + " ".join([w for w in best_rv if w not in exclude_tokens]), ofn)
    perf = corpus_eval(gen_rsp, truth_rsp)
    elapsed = time.time() - start
    info = "Full eval result {} elapsed {}".format(str(perf), elapsed)
    print(info)
    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
    return perf[params.eval_metric]


def run_s2s_dv_epoch(model, loader, criterion_sv, criterion_dv,
                     curr_epoch=0, max_grad_norm=5.0, optimizer=None, desc="Train",
                     pad_idx=0, model_name="s2s_dv", logs_dir=""):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        sv_probs, dv_probs, dv_gates = model(batch)
        sv_targets = batch[DK_TGT_SV_WID].to(device())
        dv_targets = batch[DK_TGT_DV_WID].to(device())
        dv_gate_targets = batch[DK_TGT_DV_GATE].to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        sv_log_probs, dv_log_probs = get_sv_dv_log_probs(sv_probs, dv_probs, dv_gates)
        sv_log_probs = sv_log_probs * ((1 - dv_gate_targets).unsqueeze(2).expand_as(sv_log_probs))
        dv_log_probs = dv_log_probs * (dv_gate_targets.unsqueeze(2).expand_as(dv_log_probs))
        sv_log_probs = sv_log_probs.view(-1, sv_log_probs.size(-1))
        dv_log_probs = dv_log_probs.view(-1, dv_log_probs.size(-1))
        g_loss = criterion_sv(sv_log_probs, sv_targets.contiguous().view(-1))
        c_loss = criterion_dv(dv_log_probs, dv_targets.contiguous().view(-1))
        loss = g_loss + c_loss
        total_loss += loss.item()
        total_tokens += n_tokens
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad,model.parameters()), max_grad_norm)
            optimizer.step()

        # compute acc
        tgt = copy.deepcopy(sv_targets.view(-1, 1).squeeze(1))
        dv_gate_truth = dv_gate_targets.view(-1, 1).squeeze(1)
        dv_tgt = dv_targets.view(-1, 1).squeeze(1)
        sv_preds_i = copy.deepcopy(sv_log_probs.max(1)[1])
        dv_preds_i = dv_log_probs.max(1)[1]
        sv_preds_v = sv_log_probs.max(1)[0]
        for i in range(sv_preds_i.shape[0]):
            if sv_preds_v[i] == 0:
                sv_preds_i[i] = dv_preds_i[i]
        for i in range(tgt.shape[0]):
            if dv_gate_truth[i] == 1:
                tgt[i] = dv_tgt[i]
        n_correct = sv_preds_i.data.eq(tgt.data)
        n_correct = n_correct.masked_select(tgt.ne(pad_idx).data).sum()
        total_correct += n_correct.item()

    loss_report = total_loss / total_tokens
    acc = total_correct / total_tokens
    elapsed = time.time() - start
    info = desc + " epoch %d loss %f, acc %f ppl %f elapsed time %f" % (curr_epoch, loss_report, acc,
                                                                        math.exp(loss_report), elapsed)
    print(info)
    write_line_to_file(info, logs_dir + model_name + "_train_info.txt")
    return loss_report, acc


def train_s2s_dv(params, model, train_loader, criterion_sv, criterion_dv, optimizer,
                 completed_epochs=0, eval_loader=None, best_eval_result=0, best_eval_epoch=0,
                 past_eval_results=[], checkpoint=True):
    model = model.to(device())
    criterion_sv = criterion_sv.to(device())
    criterion_dv = criterion_dv.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_s2s_dv_epoch(model, train_loader, criterion_sv, criterion_dv,
                                         curr_epoch=report_epoch, optimizer=optimizer,
                                         max_grad_norm=params.max_gradient_norm,
                                         desc="Train", pad_idx=params.pad_idx,
                                         model_name=params.model_name,
                                         logs_dir=params.logs_dir)
        if params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
            optimizer.update_learning_rate(train_loss, "max")

        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        if checkpoint:
            model_checkpoint(fn, report_epoch, model, optimizer, params,
                             past_eval_results, best_eval_result, best_eval_epoch)

        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0:
                    eval_score = eval_s2s_dv(model, eval_loader, params)
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

        print("")
    return best_eval_result, best_eval_epoch
