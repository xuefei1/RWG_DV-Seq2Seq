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
from components import SimpleRNN, Generator, AttnDecoderRNN
from utils.misc_utils import time_checkpoint


def build_strict_tgt_w2c(data_seg_list, w2v, w2c_limit=None):
    w2c = {}
    rev_map = {}
    for d in data_seg_list:
        key = " ".join(d[1])
        if key not in rev_map: rev_map[key] = []
        rev_map[key].append(d[0])
    for rep, qs in rev_map.items():
        dup_words = set()
        rep = rep.split(" ")
        for rwi, w in enumerate(rep):
            w_found = all( [w in q for q in qs] )
            if w_found:
                dup_words.add(w)
        for w in rep:
            if w not in dup_words and w in w2v:
                if w not in w2c: w2c[w] = 0
                w2c[w] += 1
    if w2c_limit is not None and len(w2c) > w2c_limit:
        tmp = sorted([(w,c) for w,c in w2c.items()],key=lambda t:t[1],reverse=True)[:w2c_limit]
        w2c = {t[0]:t[1] for t in tmp}
    print("size of strict tgt w2c {}".format(len(w2c)))
    return w2c


class Seq2Seq(nn.Module):

    def __init__(self, params, src_embed, target_word_embed, encoder, decoder, generator, dropout_prob=0.0):
        super(Seq2Seq, self).__init__()
        self.params = params
        self.src_embed = parallel(src_embed)
        self.target_word_embed = parallel(target_word_embed)
        self.encoder = encoder
        self.decoder = decoder
        self.generator = parallel(generator)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, data_dict):
        tgt = data_dict[DK_TGT_WID].to(device())
        target_len = tgt.shape[1]
        encoder_op, encoder_hidden = self.encode(data_dict)
        decoder_hidden = self.prep_enc_hidden_for_dec(encoder_hidden)
        use_teacher_forcing = True if random.random() < self.params.s2s_teacher_forcing_ratio else False
        g_probs = self.decode(decoder_hidden, encoder_op.shape[0], target_len, tgt=tgt if use_teacher_forcing else None)
        return g_probs

    def encode(self, data_dict):
        src_wid = data_dict[DK_SRC_WID].to(device())
        src_oov_wid = data_dict[DK_SRC_OOV_WID].to(device())
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        src_lens = torch.sum(src_mask.squeeze(1), dim=1)
        src = self.src_embed(src_wid, src_oov_wid)
        encoder_op, encoder_hidden = self.encoder(src, lens=src_lens)
        return encoder_op, encoder_hidden

    def prep_enc_hidden_for_dec(self, encoder_hidden):
        # TODO: does not work if the number of rnn layers is more than 1
        decoder_hidden = encoder_hidden
        if self.params.s2s_encoder_type.lower() == "lstm":
            if self.params.s2s_decoder_type.lower() == "lstm":
                if self.params.s2s_encoder_rnn_dir == 2:
                    h = torch.cat([encoder_hidden[0][0, :, :], encoder_hidden[0][1, :, :]], dim=1).unsqueeze(0)
                    c = torch.cat([encoder_hidden[1][0, :, :], encoder_hidden[1][1, :, :]], dim=1).unsqueeze(0)
                else:
                    h = encoder_hidden[0]
                    c = encoder_hidden[1]
                decoder_hidden = (h, c)
            else:
                if self.params.s2s_encoder_rnn_dir == 2:
                    decoder_hidden = torch.cat([encoder_hidden[0][0, :, :], encoder_hidden[0][1, :, :]], dim=1).unsqueeze(0)
                else:
                    decoder_hidden = encoder_hidden[0]
        elif self.params.s2s_encoder_type.lower() == "gru":
            if self.params.s2s_decoder_type.lower() == "lstm":
                if self.params.s2s_encoder_rnn_dir == 2:
                    h = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
                    c = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
                else:
                    h = encoder_hidden
                    c = encoder_hidden
                decoder_hidden = (h, c)
            else:
                if self.params.s2s_encoder_rnn_dir == 2:
                    decoder_hidden = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=1).unsqueeze(0)
                else:
                    decoder_hidden = encoder_hidden
        return decoder_hidden # B x E

    def decode_step(self, decoder_input, decoder_hidden):
        decoder_input = self.target_word_embed(decoder_input)
        decoder_output, hidden = self.decoder(decoder_input, decoder_hidden)
        decoder_output = self.dropout(decoder_output)
        probs = self.generator(decoder_output)
        return probs, hidden

    def decode(self, dec_hidden, batch_size, target_length, tgt=None):
        sos = torch.ones(batch_size,1).fill_(self.params.sos_idx).long().to(device())
        gen_probs = []
        dec_input = sos
        for di in range(target_length):
            probs, dec_hidden = self.decode_step(dec_input, dec_hidden)
            if tgt is not None:
                dec_input = tgt[:, di].unsqueeze(1)
            else:
                _, next_wi = probs.topk(1)
                dec_input = next_wi.squeeze(2).detach().long().to(device())
            gen_probs.append(probs)
        return torch.cat(gen_probs, dim=1)


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
        # context = attn_weights.bmm(encoder_outputs)
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


class ConcatAttention(nn.Module):

    def __init__(self, attend_dim, query_dim, att_dim):
        super(ConcatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        init_weights(self)

    def forward(self, input, context, precompute, mask=None):
        if precompute is None:
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
        weightedContext = torch.bmm(score_m, context)  # batch x 1 x dim
        return weightedContext, score_m, precompute


class MaxOut(nn.Module):

    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
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


class ACMDecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, tgt_vocab_size, n_layers=1, rnn_type="gru", max_pool_size=2, dropout_prob=0.1):
        super(ACMDecoderRNN, self).__init__()
        self.max_pool_size = max_pool_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.concat_compression = parallel(nn.Linear(self.hidden_size * 2 + self.input_size, self.hidden_size))
        self.max_out = MaxOut(self.max_pool_size)
        self.attn = parallel(ConcatAttention(self.hidden_size, self.hidden_size, self.hidden_size))
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
        context, attn_weights, precomp = self.attn(rnn_output, encoder_outputs, precomp, mask=batch_enc_attn_mask)
        # context = attn_weights.bmm(encoder_outputs)
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


def get_gen_cpy_log_probs(g_wid_probs, c_wid_probs, c_gate_probs):
    c_wid_probs = c_wid_probs * c_gate_probs.expand_as(c_wid_probs) + 1e-8
    g_wid_probs = g_wid_probs * (1 - c_gate_probs).expand_as(g_wid_probs) + 1e-8
    c_log_wid_probs = torch.log(c_wid_probs)
    g_log_wid_probs = torch.log(g_wid_probs)
    return g_log_wid_probs, c_log_wid_probs


class Seq2SeqAttn(nn.Module):

    def __init__(self, params, src_embed, target_word_embed, encoder, decoder, generator):
        super(Seq2SeqAttn, self).__init__()
        self.params = params
        self.src_embed = parallel(src_embed)
        self.target_word_embed = parallel(target_word_embed)
        self.encoder = encoder
        self.decoder = decoder
        self.generator = parallel(generator)
        self.hidden_ff = parallel(nn.Linear(encoder.hidden_size * encoder.rnn_dir, encoder.hidden_size * encoder.rnn_dir))

    def forward(self, data_dict, rv_data={}):
        tgt = data_dict[DK_TGT_GEN_WID].to(device())
        src_mask = data_dict[DK_SRC_WID_MASK].to(device())
        target_len = tgt.shape[1]
        encoder_op, encoder_hidden = self.encode(data_dict)
        decoder_hidden = self.prep_enc_hidden_for_dec(encoder_hidden)
        use_teacher_forcing = True if random.random() < self.params.s2s_teacher_forcing_ratio else False
        g_probs, attns = self.decode(encoder_op, decoder_hidden, src_mask, target_len,
                                     tgt=tgt if use_teacher_forcing else None)
        rv_data["attn_weights"] = attns
        return g_probs

    def encode(self, data_dict):
        src = data_dict[DK_SRC_WID].to(device())
        src_mask = data_dict[DK_SRC_WID_MASK]
        src_lens = torch.sum(src_mask.squeeze(1), dim=1)
        src = self.src_embed(src)
        encoder_op, encoder_hidden = self.encoder(src, lens=src_lens)
        return encoder_op, encoder_hidden

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

    def decode_step(self, decoder_input, decoder_hidden, encoder_past_hs, src_mask=None):
        decoder_input = self.target_word_embed(decoder_input)
        output, hidden, attn_ws = self.decoder(decoder_input, decoder_hidden, encoder_past_hs,
                                               batch_enc_attn_mask=src_mask)
        output = self.generator(output)
        return output, hidden, attn_ws

    def decode(self, enc_hs, dec_hidden, src_mask, target_length, tgt=None):
        sos = torch.ones(enc_hs.shape[0],1).fill_(self.params.sos_idx).long().to(device())
        gen_probs = []
        dec_attns = []
        dec_input = sos
        for di in range(target_length):
            dec_output, dec_hidden, dec_attn = self.decode_step(dec_input, dec_hidden, enc_hs, src_mask)
            if tgt is not None:
                dec_input = tgt[:, di].unsqueeze(1)
            else:
                _, next_wi = dec_output.topk(1)
                dec_input = next_wi.squeeze(2).detach().long().to(device())
            gen_probs.append(dec_output)
            dec_attns.append(dec_attn)
        return torch.cat(gen_probs, dim=1), torch.cat(dec_attns, dim=1)


class Seq2SeqACM(nn.Module):

    def __init__(self, params, src_embed, target_word_embed, encoder, decoder):
        super(Seq2SeqACM, self).__init__()
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
        src_mask = data_dict[DK_SRC_WID_MASK]
        encoder_hidden = None
        encoder_cell = None
        src_lens = torch.sum(src_mask.squeeze(1), dim=1)
        if self.params.s2s_encoder_type.lower() == "lstm": encoder_hidden = (encoder_hidden, encoder_cell)
        src = self.src_embed(data_dict)
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


class QryDataEmbedding(nn.Module):

    def __init__(self, d_model, word_embed, resize=False):
        super(QryDataEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embed = word_embed
        self.resize_layer = nn.Linear(word_embed.d_model, self.d_model) if resize else None

    def forward(self, data_dict):
        w_id = data_dict[DK_SRC_WID].to(device())
        w_embed = self.word_embed(w_id)
        if self.resize_layer is not None:
            w_embed = self.resize_layer(w_embed)
        assert w_embed.shape[2] == self.d_model, "Embedding size does not match expect encoder input size"
        return w_embed


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


class S2SBeamCandidate:

    def __init__(self, curr_word_index_tsr, score, curr_decoded_words, decoder_hidden, context=None):
        self.curr_word_index_tsr = curr_word_index_tsr
        self.score = score
        self.curr_decoded_words = curr_decoded_words
        self.decoder_hidden = decoder_hidden
        self.context = context

    def __repr__(self):
        return str(self.curr_decoded_words)

    def __str__(self):
        return str(self.curr_decoded_words)


class S2SBeamInstance:

    def __init__(self, enc_h, i2w, idx_in_batch, src_seg_list,
                 beam_width=4, sos_idx=2, eos_idx=3, ctx=None,
                 gamma=0.0, len_norm=0.0):
        self.i2w = i2w
        self.idx_in_batch = idx_in_batch
        self.beam_width = beam_width
        self.gamma = gamma
        self.len_norm = len_norm
        self.eos_idx = eos_idx
        self.src_seg_list = src_seg_list
        sos = torch.ones(1,1).fill_(sos_idx).long().to(device())
        self.curr_candidates = [
            S2SBeamCandidate(curr_word_index_tsr=sos, score=0.0, curr_decoded_words=[i2w[sos_idx]],
                             decoder_hidden=enc_h, context=ctx)
        ]
        self.completed_insts = []
        self.all_done = False

    def norm_cand_score_by_len(self, cand):
        new_score = float('-inf')
        content_len = len([w for w in cand.curr_decoded_words if w not in {SOS_TOKEN, EOS_TOKEN, PAD_TOKEN}])
        if content_len > 0:
            if self.len_norm > 0:
                length_penalty = (self.len_norm + content_len) / (self.len_norm + 1)
                new_score /= length_penalty ** self.len_norm
            else:
                new_score = new_score / content_len
        cand.score = new_score

    def update(self, next_vals, next_wids, dec_hs, ctx=None):
        assert dec_hs.shape[0] == len(self.curr_candidates)
        next_vals = next_vals.squeeze(1)
        next_wids = next_wids.squeeze(1)
        next_candidates = []
        for cand_i, cand in enumerate(self.curr_candidates):
            decoder_hidden = dec_hs[cand_i, :].unsqueeze(0)
            context = ctx[cand_i, :].unsqueeze(0) if ctx is not None else None
            for i in range(next_wids[cand_i, :].shape[0]):
                pred_word_idx = next_wids[cand_i, i].item()
                div_penalty = 0.0
                if i > 0: div_penalty = self.gamma * (i + 1)
                new_score = cand.score + next_vals[cand_i, i].item() - div_penalty
                new_tgt = torch.ones(1,1).long().fill_(pred_word_idx).to(device())
                new_word = self.i2w[pred_word_idx] if pred_word_idx in self.i2w else OOV_TOKEN
                new_cand = S2SBeamCandidate(curr_word_index_tsr=new_tgt,
                                            score=new_score,
                                            curr_decoded_words=[w for w in cand.curr_decoded_words],
                                            decoder_hidden=decoder_hidden,
                                            context=context)
                new_cand.curr_decoded_words.append(new_word)
                if pred_word_idx == self.eos_idx:
                    self.norm_cand_score_by_len(new_cand)
                    self.completed_insts.append(new_cand)
                else:
                    next_candidates.append(new_cand)
        if len(next_candidates) > self.beam_width:
            next_candidates = sorted(next_candidates, key=lambda c:c.score, reverse=True)
            next_candidates = next_candidates[:self.beam_width]
        self.curr_candidates = next_candidates
        self.all_done = len(self.curr_candidates) == 0

    def get_curr_tgt(self):
        if len(self.curr_candidates) == 0:
            assert False, "No available candidate found when calling get_curr_tgt()"
        return torch.cat([c.curr_word_index_tsr for c in self.curr_candidates], dim=0).long().to(device())

    def get_curr_dec_hidden(self):
        if len(self.curr_candidates) == 0:
            assert False, "No available candidate found when calling get_curr_dec_hidden()"
        return torch.cat([c.decoder_hidden for c in self.curr_candidates], dim=1).float().to(device())

    def get_curr_context(self):
        if len(self.curr_candidates) == 0:
            assert False, "No available candidate found when calling get_curr_context()"
        return torch.cat([c.context for c in self.curr_candidates if c is not None], dim=0).float().to(device())

    def get_curr_candidate_size(self):
        return len(self.curr_candidates)

    def get_curr_src_seg_list(self):
        return [self.src_seg_list for _ in range(len(self.curr_candidates))]

    def collect_results(self, topk=1):
        self.completed_insts = sorted(self.completed_insts, key=lambda c: c.score, reverse=True)
        if len(self.curr_candidates) > 1:
            for cand in self.curr_candidates:
                self.norm_cand_score_by_len(cand)
            self.curr_candidates = sorted(self.curr_candidates, key=lambda c: c.score, reverse=True)
        for cand in self.curr_candidates:
            self.completed_insts.append(cand)
        return self.completed_insts[:topk]


def make_s2s_qrw_model(src_w2v_mat, tgt_w2v_mat, params, src_vocab_size, tgt_vocab_size,
                       same_word_embedding=False, pad_idx=0):
    if src_w2v_mat is None:
        src_w_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size, pad_idx)
    else:
        src_w_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                    allow_further_training=params.word_embed_further_training)
    if tgt_w2v_mat is None:
        tgt_w_embed_layer = TrainableEmbedding(params.word_embedding_dim, tgt_vocab_size, pad_idx)
    else:
        tgt_w_embed_layer = PreTrainedWordEmbedding(tgt_w2v_mat, params.word_embedding_dim,
                                                    allow_further_training=params.word_embed_further_training)
    if same_word_embedding and src_vocab_size == tgt_vocab_size:
        tgt_w_embed_layer = src_w_embed_layer
    encoder = SimpleRNN(params.word_embedding_dim, params.s2s_encoder_hidden_size,
                        dropout_prob=params.s2s_encoder_dropout_prob, rnn_type=params.s2s_encoder_type,
                        rnn_dir=params.s2s_encoder_rnn_dir, n_layers=params.s2s_num_encoder_layers)
    decoder = SimpleRNN(params.word_embedding_dim, params.s2s_encoder_hidden_size * params.s2s_encoder_rnn_dir,
                        dropout_prob=params.s2s_decoder_dropout_prob, n_layers=params.s2s_num_decoder_layers,
                        rnn_type=params.s2s_decoder_type, rnn_dir=1)
    generator = Generator(params.s2s_encoder_hidden_size * params.s2s_encoder_rnn_dir, tgt_vocab_size)
    model = Seq2Seq(params, src_w_embed_layer, tgt_w_embed_layer, encoder, decoder, generator,
                    dropout_prob=params.s2s_model_dropout_prob)
    return model


def make_s2s_attn_qrw_model(src_w2v_mat, tgt_w2v_mat, params, src_vocab_size, tgt_vocab_size,
                            same_word_embedding=False, pad_idx=0):

    if src_w2v_mat is None:
        src_w_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size, pad_idx)
    else:
        src_w_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                    allow_further_training=params.word_embed_further_training)
    if tgt_w2v_mat is None:
        tgt_w_embed_layer = TrainableEmbedding(params.word_embedding_dim, tgt_vocab_size, pad_idx)
    else:
        tgt_w_embed_layer = PreTrainedWordEmbedding(tgt_w2v_mat, params.word_embedding_dim,
                                                    allow_further_training=params.word_embed_further_training)
    if same_word_embedding and src_vocab_size == tgt_vocab_size:
        tgt_w_embed_layer = src_w_embed_layer
    encoder = SimpleRNN(params.word_embedding_dim, params.s2s_encoder_hidden_size,
                        dropout_prob=params.s2s_encoder_dropout_prob, rnn_type=params.s2s_encoder_type,
                        rnn_dir=params.s2s_encoder_rnn_dir, n_layers=params.s2s_num_encoder_layers)
    decoder = AttnDecoderRNN(params.word_embedding_dim, params.s2s_encoder_hidden_size * params.s2s_encoder_rnn_dir,
                             dropout_prob=params.s2s_decoder_dropout_prob, n_layers=params.s2s_num_decoder_layers,
                             rnn_type=params.s2s_decoder_type)
    generator = Generator(params.s2s_encoder_hidden_size * params.s2s_encoder_rnn_dir, tgt_vocab_size)
    model = Seq2SeqAttn(params, src_w_embed_layer, tgt_w_embed_layer, encoder, decoder, generator)
    return model


def make_s2s_acm_qrw_model(src_w2v_mat, tgt_w2v_mat, params, src_vocab_size, tgt_vocab_size,
                           same_word_embedding=False, pad_idx=0):
    if src_w2v_mat is None:
        src_w_embed_layer = TrainableEmbedding(params.word_embedding_dim, src_vocab_size, pad_idx)
    else:
        src_w_embed_layer = PreTrainedWordEmbedding(src_w2v_mat, params.word_embedding_dim,
                                                    allow_further_training=params.word_embed_further_training)
    if tgt_w2v_mat is None:
        tgt_w_embed_layer = TrainableEmbedding(params.word_embedding_dim, tgt_vocab_size, pad_idx)
    else:
        tgt_w_embed_layer = PreTrainedWordEmbedding(tgt_w2v_mat, params.word_embedding_dim,
                                                    allow_further_training=params.word_embed_further_training)
    if same_word_embedding and src_vocab_size == tgt_vocab_size:
        tgt_w_embed_layer = src_w_embed_layer
    src_embed_layer = QryDataEmbedding(params.word_embedding_dim, src_w_embed_layer, resize=False)
    encoder = SimpleRNN(params.word_embedding_dim, params.s2s_encoder_hidden_size,
                        dropout_prob=params.s2s_encoder_dropout_prob, rnn_type=params.s2s_encoder_type,
                        rnn_dir=params.s2s_encoder_rnn_dir, n_layers=params.s2s_num_encoder_layers)
    decoder = ACMDecoderRNN(params.word_embedding_dim, params.s2s_encoder_hidden_size * params.s2s_encoder_rnn_dir,
                            tgt_vocab_size, n_layers=params.s2s_num_decoder_layers, rnn_type=params.s2s_decoder_type,
                            dropout_prob=params.s2s_decoder_dropout_prob)
    model = Seq2SeqACM(params, src_embed_layer, tgt_w_embed_layer, encoder, decoder)
    return model


def full_eval_s2s(model, loader, params, desc="Full eval"):
    start = time.time()
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    truth_rsp = []
    gen_rsp = []
    ofn = params.logs_dir + params.model_name + "_full_eval_out.txt"
    # write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        beam_rvs = s2s_beam_decode_batch(model, batch, params.sos_idx, params.tgt_i2w, eos_idx=params.eos_idx,
                                         len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                         max_len=params.max_decoded_seq_len, beam_width=params.beam_width_eval)
        for bi in range(batch[DK_BATCH_SIZE]):
            best_rv = beam_rvs[bi][0].curr_decoded_words
            truth_rsp.append([[w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]])
            gen_rsp.append([w for w in best_rv if w not in exclude_tokens])
            # write_line_to_file("truth: " + " ".join([w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]), ofn)
            # write_line_to_file("pred: " + " ".join([w for w in best_rv if w not in exclude_tokens]), ofn)
    perf = corpus_eval(gen_rsp, truth_rsp)
    elapsed = time.time() - start
    info = "Full eval result {} elapsed {}".format(str(perf), elapsed)
    print(info)
    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
    return perf[params.eval_metric]


def full_eval_s2s_attn(model, loader, params, desc="Full eval"):
    start = time.time()
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    truth_rsp = []
    gen_rsp = []
    ofn = params.logs_dir + params.model_name + "_full_eval_out.txt"
    # write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        batch = copy.deepcopy(batch)
        beam_rvs = s2s_attn_beam_decode_batch(model, batch, params.sos_idx, params.tgt_i2w, eos_idx=params.eos_idx,
                                              len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                              max_len=params.max_decoded_seq_len, beam_width=params.beam_width_eval)
        for bi in range(batch[DK_BATCH_SIZE]):
            best_rv = beam_rvs[bi][3]
            truth_rsp.append([[w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]])
            gen_rsp.append([w for w in best_rv if w not in exclude_tokens])
            # write_line_to_file("truth: " + " ".join([w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]), ofn)
            # write_line_to_file("pred: " + " ".join([w for w in best_rv if w not in exclude_tokens]), ofn)
    perf = corpus_eval(gen_rsp, truth_rsp)
    elapsed = time.time() - start
    info = "Full eval result {} elapsed {}".format(str(perf), elapsed)
    print(info)
    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
    return perf[params.eval_metric]


def full_eval_s2s_acm(model, loader, params, desc="Full eval"):
    start = time.time()
    exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ""]
    truth_rsp = []
    gen_rsp = []
    ofn = params.logs_dir + params.model_name + "_full_eval_out.txt"
    # write_line_to_file("", ofn)
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        batch = copy.deepcopy(batch)
        beam_rvs = s2s_acm_beam_decode_batch(model, batch, params.sos_idx, params.tgt_i2w, eos_idx=params.eos_idx,
                                             len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                             max_len=params.max_decoded_seq_len, beam_width=params.beam_width_eval)
        for bi in range(batch[DK_BATCH_SIZE]):
            best_rv = beam_rvs[bi][0][3]
            truth_rsp.append([[w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]])
            gen_rsp.append([w for w in best_rv if w not in exclude_tokens])
            # write_line_to_file("truth: " + " ".join([w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]), ofn)
            # write_line_to_file("pred: " + " ".join([w for w in best_rv if w not in exclude_tokens]), ofn)
    perf = corpus_eval(gen_rsp, truth_rsp)
    elapsed = time.time() - start
    info = "Full eval result {} elapsed {}".format(str(perf), elapsed)
    print(info)
    write_line_to_file(info, params.logs_dir + params.model_name + "_train_info.txt")
    return perf[params.eval_metric]


def run_s2s_epoch(model, loader, criterion, curr_epoch=0, max_grad_norm=5.0, optimizer=None,
                  desc="Train", pad_idx=0, model_name="s2s", logs_dir=""):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        g_log_wid_probs = model(batch)
        gen_targets = batch[DK_TGT_WID].to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        g_log_wid_probs = g_log_wid_probs.view(-1, g_log_wid_probs.size(-1))
        loss = criterion(g_log_wid_probs, gen_targets.contiguous().view(-1))
        # compute acc
        tgt = copy.deepcopy(gen_targets.view(-1, 1).squeeze(1))
        g_preds_i = copy.deepcopy(g_log_wid_probs.max(1)[1])
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


def run_s2s_attn_epoch(model, loader, criterion, curr_epoch=0, max_grad_norm=5.0, optimizer=None,
                       desc="Train", pad_idx=0, model_name="s2s_attn", logs_dir=""):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0
    for batch in tqdm(loader, mininterval=2, desc=desc, leave=False, ascii=True):
        g_log_wid_probs = model(batch)
        gen_targets = batch[DK_TGT_GEN_WID].to(device())
        n_tokens = batch[DK_TGT_N_TOKENS].item()
        g_log_wid_probs = g_log_wid_probs.view(-1, g_log_wid_probs.size(-1))
        loss = criterion(g_log_wid_probs, gen_targets.contiguous().view(-1))
        # compute acc
        tgt = copy.deepcopy(gen_targets.view(-1, 1).squeeze(1))
        g_preds_i = copy.deepcopy(g_log_wid_probs.max(1)[1])
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


def run_s2s_acm_epoch(model, loader, criterion_gen, criterion_cpy,
                      curr_epoch=0, max_grad_norm=5.0, optimizer=None,
                      desc="Train", pad_idx=0, model_name="s2s_acm",
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


def train_s2s(params, model, train_loader, criterion, optimizer,
              completed_epochs=0, eval_loader=None, checkpoint=True,
              best_eval_result=0, best_eval_epoch=0, past_eval_results=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_s2s_epoch(model, train_loader, criterion, curr_epoch=report_epoch, optimizer=optimizer,
                                      max_grad_norm=params.max_gradient_norm, desc="Train", pad_idx=params.pad_idx,
                                      model_name=params.model_name, logs_dir=params.logs_dir)
        if params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
            optimizer.update_learning_rate(train_loss, "max")
        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0:
                    eval_score = full_eval_s2s(model, eval_loader, params)
                    if eval_score > best_eval_result:
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
                    if not params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
                        optimizer.update_learning_rate(eval_score, "min")
                    past_eval_results.append(eval_score)
                    if len(past_eval_results) > params.past_eval_scores_considered:
                        past_eval_results = past_eval_results[1:]
        fn = params.saved_models_dir + params.model_name + "_latest.pt"
        if checkpoint:
            model_checkpoint(fn, report_epoch, model, optimizer, params,
                             past_eval_results, best_eval_result, best_eval_epoch)
        print("")
    return best_eval_result, best_eval_epoch


def train_s2s_attn(params, model, train_loader, criterion, optimizer,
                   completed_epochs=0, eval_loader=None, checkpoint=True,
                   best_eval_result=0, best_eval_epoch=0, past_eval_results=[]):
    model = model.to(device())
    criterion = criterion.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_s2s_attn_epoch(model, train_loader, criterion,
                                           curr_epoch=report_epoch, optimizer=optimizer,
                                           max_grad_norm=params.max_gradient_norm,
                                           desc="Train", pad_idx=params.pad_idx,
                                           model_name=params.model_name, logs_dir=params.logs_dir)
        if params.lr_decay_with_train_perf and hasattr(optimizer, "update_learning_rate"):
            optimizer.update_learning_rate(train_loss, "max")
        if eval_loader is not None:
            model.eval()
            with torch.no_grad():
                if report_epoch >= params.full_eval_start_epoch and \
                   report_epoch % params.full_eval_every_epoch == 0: # full eval
                    eval_score = full_eval_s2s_attn(model, eval_loader, params)
                    if eval_score > best_eval_result:
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


def train_s2s_acm(params, model, train_loader, criterion_gen, criterion_cpy, optimizer,
                  completed_epochs=0, eval_loader=None, best_eval_result=0, best_eval_epoch=0,
                  past_eval_results=[], checkpoint=True):
    model = model.to(device())
    criterion_gen = criterion_gen.to(device())
    criterion_cpy = criterion_cpy.to(device())
    for epoch in range(params.epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()
        train_loss, _ = run_s2s_acm_epoch(model, train_loader, criterion_gen, criterion_cpy,
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
                    eval_score = full_eval_s2s_acm(model, eval_loader, params)
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


def s2s_beam_decode_batch(model, batch, start_idx, i2w, max_len,
                          gamma=0.0, beam_width=4, eos_idx=3,
                          len_norm=0.0, topk=1):
    batch_size = batch[DK_SRC_WID].shape[0]
    model = model.to(device())
    encoder_op, encoder_hidden = model.encode(batch)
    encoder_hidden = model.prep_enc_hidden_for_dec(encoder_hidden)
    batch_results = [
        S2SBeamInstance(idx_in_batch=bi, i2w=i2w, src_seg_list=batch[DK_SRC_SEG_LISTS][bi],
                        enc_h=encoder_hidden[:, bi, :].unsqueeze(0),
                        beam_width=beam_width, sos_idx=start_idx,
                        eos_idx=eos_idx, ctx=None,
                        gamma=gamma, len_norm=len_norm)
        for bi in range(batch_size)
    ]
    final_ans = []
    for i in range(max_len):
        curr_actives = [b for b in batch_results if not b.all_done]
        if len(curr_actives) == 0: break
        b_tgt_list = [b.get_curr_tgt() for b in curr_actives]
        b_tgt = torch.cat(b_tgt_list, dim=0)
        b_hidden_list = [b.get_curr_dec_hidden() for b in curr_actives]
        b_hidden = torch.cat(b_hidden_list, dim=1).to(device())
        b_cand_size_list = [b.get_curr_candidate_size() for b in curr_actives]
        gen_wid_probs, b_hidden = model.decode_step(b_tgt, b_hidden)
        beam_i = 0
        for bi, size in enumerate(b_cand_size_list):
            g_probs = gen_wid_probs[beam_i:beam_i + size, :].view(size, -1, gen_wid_probs.size(-1))
            hiddens = b_hidden[:, beam_i:beam_i + size, :].view(size, -1, b_hidden.size(-1))
            cand_vals, cand_inds = g_probs.topk(beam_width)
            curr_actives[bi].update(cand_vals, cand_inds, hiddens)
            beam_i += size
    for b in batch_results:
        final_ans.append(b.collect_results(topk=topk))
    return final_ans


def s2s_attn_beam_decode_batch(model, batch, start_idx, i2w, max_len,
                               gamma=0.0, beam_width=4, eos_idx=3, len_norm=1.0, topk=1):
    batch_size = batch[DK_SRC_WID].shape[0]
    model = model.to(device())
    src_mask = batch[DK_SRC_WID_MASK].to(device())
    encoder_op, encoder_hidden = model.encode(batch)
    encoder_hidden = model.prep_enc_hidden_for_dec(encoder_hidden)
    batch_results = [
        S2SBeamSearchResult(idx_in_batch=bi, i2w=i2w, src_seg_list=batch[DK_SRC_SEG_LISTS][bi],
                            enc_h=encoder_hidden[:, bi, :].unsqueeze(0), beam_width=beam_width, sos_idx=start_idx,
                            eos_idx=eos_idx, ctx=None, gamma=gamma, len_norm=len_norm)
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
        b_cand_size_list = [b.get_curr_candidate_size() for b in curr_actives]
        enc_op = torch.cat([encoder_op[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in curr_actives], dim=0)
        s_mask = torch.cat([src_mask[b.idx_in_batch, :, :].unsqueeze(0).repeat(b.get_curr_candidate_size(), 1, 1) for b in curr_actives], dim=0)
        gen_wid_probs, b_hidden, _ = model.decode_step(b_tgt, b_hidden, enc_op, s_mask)
        beam_i = 0
        for bi, size in enumerate(b_cand_size_list):
            g_probs = gen_wid_probs[beam_i:beam_i+size, :].view(size, -1, gen_wid_probs.size(-1))
            hiddens = b_hidden[:, beam_i:beam_i+size, :].view(size, -1, b_hidden.size(-1))
            vt, it = g_probs.topk(beam_width)
            next_vals, next_wids, next_words, dec_hs = [], [], [], []
            for ci in range(size):
                vals, wis, words = [], [], []
                for idx in range(beam_width):
                    vals.append(vt[ci,0,idx].item())
                    wi = it[ci,0,idx].item()
                    word = i2w[wi]
                    wis.append(wi)
                    words.append(word)
                next_vals.append(vals)
                next_wids.append(wis)
                next_words.append(words)
            dec_hs = [hiddens[j,:].unsqueeze(1) for j in range(hiddens.shape[0])]
            curr_actives[bi].update(g_probs, next_vals, next_wids, next_words, dec_hs)
            beam_i += size
    for b in batch_results:
        final_ans += b.collect_results(topk=topk)
    return final_ans


def s2s_acm_beam_decode_batch(model, batch, start_idx, i2w, max_len,
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


def unit_update(batch_idx, cand_size, gen_wid_probs, b_hidden, curr_beam_width, curr_actives):
    g_probs = gen_wid_probs[batch_idx]
    hiddens = b_hidden[batch_idx]
    cand_vals, cand_inds = g_probs.topk(curr_beam_width)
    curr_actives[batch_idx].update(cand_vals, cand_inds, hiddens)


def parallel_update(b_cand_size_list, g_probs, hidden, curr_beam_width, curr_actives):
    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(unit_update)(bi, size, g_probs, hidden, curr_beam_width, curr_actives) for bi, size in enumerate(b_cand_size_list))


def s2s_dynamic_beam_decode_batch(model, batch, start_idx, i2w, max_len,
                                  gamma=0.0, beam_width_list=[4], eos_idx=3,
                                  len_norm=0.0, topk=1):
    # start_all = time_checkpoint()
    batch_size = batch[DK_SRC_WID].shape[0]
    model = model.to(device())
    encoder_op, encoder_hidden = model.encode(batch)
    encoder_hidden = model.prep_enc_hidden_for_dec(encoder_hidden)
    batch_results = [
        S2SBeamInstance(idx_in_batch=bi, i2w=i2w, src_seg_list=batch[DK_SRC_SEG_LISTS][bi],
                        enc_h=encoder_hidden[:, bi, :].unsqueeze(0),
                        beam_width=max(beam_width_list), sos_idx=start_idx,
                        eos_idx=eos_idx, ctx=None,
                        gamma=gamma, len_norm=len_norm)
        for bi in range(batch_size)
    ]
    final_ans = []
    # start_decode = time_checkpoint()
    for i in range(max_len):
        # start_outer = time_checkpoint()
        curr_beam_width = beam_width_list[i] if i < len(beam_width_list) else beam_width_list[-1]
        curr_actives = [b for b in batch_results if not b.all_done]
        if len(curr_actives) == 0: break
        b_tgt_list = [b.get_curr_tgt() for b in curr_actives]
        b_tgt = torch.cat(b_tgt_list, dim=0)
        b_hidden_list = [b.get_curr_dec_hidden() for b in curr_actives]
        b_hidden = torch.cat(b_hidden_list, dim=1).to(device())
        b_cand_size_list = [b.get_curr_candidate_size() for b in curr_actives]
        # assert sum(b_cand_size_list) == b_tgt.size(0)
        gen_wid_probs, b_hidden = model.decode_step(b_tgt, b_hidden)
        # start_inner = time_checkpoint()
        beam_i = 0
        for bi, size in enumerate(b_cand_size_list):
            g_probs = gen_wid_probs[beam_i:beam_i + size, :].view(size, -1, gen_wid_probs.size(-1))
            hiddens = b_hidden[:, beam_i:beam_i + size, :].view(size, -1, b_hidden.size(-1))
            cand_vals, cand_inds = g_probs.topk(curr_beam_width)
            # start_update = time_checkpoint()
            curr_actives[bi].update(cand_vals, cand_inds, hiddens)
            beam_i += size
            # time_checkpoint(start_update, "update run", verbose=bi==1)

        # g_probs_list = []
        # hiddens_list = []
        # for bi, size in enumerate(b_cand_size_list):
        #     g_probs = gen_wid_probs[beam_i:beam_i+size, :].view(size, -1, gen_wid_probs.size(-1))
        #     hiddens = b_hidden[:, beam_i:beam_i+size, :].view(size, -1, b_hidden.size(-1))
        #     g_probs_list.append(g_probs)
        #     hiddens_list.append(hiddens)
        # parallel_update(b_cand_size_list, g_probs_list, hiddens_list, curr_beam_width, curr_actives)

        # time_checkpoint(start_inner, "inner loop", verbose=i==1)
        # time_checkpoint(start_outer, "one token", verbose=i==1)
    # time_checkpoint(start_decode, "decode")
    # start_collect = time_checkpoint()
    for b in batch_results:
        final_ans.append(b.collect_results(topk=topk))
    # time_checkpoint(start_collect, "collect")
    # time_checkpoint(start_all, "complete one batch")
    return final_ans
