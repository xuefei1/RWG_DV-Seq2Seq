import copy
import math
import torch
import torch.nn as nn
import numpy as np
from constants import *
import torch.nn.functional as F
from utils.model_utils import device, init_weights


def get_nll_criterion(reduction="sum"):
    crit = nn.NLLLoss(reduction=reduction)
    return crit


def get_bce_criterion(reduction="sum"):
    crit = nn.BCELoss(reduction=reduction)
    return crit


def get_masked_nll_criterion(vocab_size, pad_idx=0, reduction="sum"):
    weight = torch.ones(vocab_size)
    weight[pad_idx] = 0
    crit = nn.NLLLoss(reduction=reduction, weight=weight)
    return crit


def get_masked_bce_criterion(vocab_size, pad_idx=0, reduction="sum"):
    weight = torch.ones(vocab_size)
    weight[pad_idx] = 0
    crit = nn.BCELoss(reduction=reduction, weight=weight)
    return crit


def make_std_mask(tgt, pad):
    if pad is not None:
        tgt_mask = (tgt != pad).unsqueeze(-2)
    else:
        tgt_mask = torch.ones(tgt.size()).type(torch.ByteTensor).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask.to(device())


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class QANetEncoderBlock(nn.Module):

    def __init__(self, norm_conv, norm_self_attn, norm_ff,
                 n_convs=1, n_attns=1, n_ffs=1):
        super(QANetEncoderBlock, self).__init__()
        self.convs = clones(norm_conv, n_convs)
        self.attns = clones(norm_self_attn, n_attns)
        self.ffs = clones(norm_ff, n_ffs)

    def forward(self, x, mask=None):
        rv = x
        for conv in self.convs:
            rv = conv(rv)
        for attn in self.attns:
            rv = attn(rv, mask)
        for ff in self.ffs:
            rv = ff(rv)
        return rv


class Attn(nn.Module):

    def __init__(self, hidden_size, multiply_outputs=False):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.multiply_outputs = multiply_outputs

    def forward(self, hidden, encoder_outputs, attn_mask=None):
        enc_seq_len = encoder_outputs.size(1)
        # For each batch of encoder outputs
        attn_energies = torch.bmm(hidden, self.attn(encoder_outputs).transpose(1,2))
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(1, attn_energies.size(1), 1).view(attn_energies.size())
            assert attn_mask.size() == attn_energies.size(), "Attention mask shape {} mismatch with Attention weights tensor shape {}.".format(attn_mask.size(), attn_energies.size())
        if attn_mask is not None and enc_seq_len > 1:
            attn_energies.data.masked_fill_(attn_mask==0, -float('inf'))
        # Normalize energies to weights in range 0 to 1
        attn_energies = F.softmax(attn_energies, dim=2)
        rv = attn_energies
        if self.multiply_outputs:
            rv = attn_energies.bmm(encoder_outputs)
        return rv


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


class AttnDecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size,
                 rnn_type="gru", n_layers=1, dropout_prob=0.0):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.concat_compression = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = Attn(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        if self.rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=False)
        else:
            self.rnn = nn.GRU(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.n_layers,
                               dropout=self.dropout_prob if self.n_layers>1 else 0,
                               batch_first=True,
                               bidirectional=False)
        init_weights(self, base_model_type="rnn")

    def forward(self, embedding, hidden, encoder_outputs, batch_enc_attn_mask=None):
        rnn_output, hidden = self.rnn(embedding, hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs, attn_mask=batch_enc_attn_mask)
        context = attn_weights.bmm(encoder_outputs)
        concat_input = torch.cat([rnn_output, context], dim=2)
        concat_output = torch.tanh(self.concat_compression(concat_input))
        output = self.dropout(concat_output)
        if torch.isnan(output).any(): assert False, "NaN detected in output when decoding with attention!"
        if torch.isnan(attn_weights).any(): assert False, "NaN detected in attn_weights when decoding with attention!"
        return output, hidden, attn_weights


class AttnRNN(nn.Module):

    def __init__(self, attn_block, input_size, hidden_size, n_layers=1, rnn_dir=2, dropout_prob=0.0, rnn_type="gru"):
        super(AttnRNN, self).__init__()
        self.attn_block = attn_block
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dir = rnn_dir
        self.rnn_type = rnn_type
        self.dropout_prob = dropout_prob
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

    def forward(self, embedded, memory, hidden=None, lens=None):
        if lens is None:
            outputs, hidden = self.rnn(embedded, hidden)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lens, batch_first=True)
            outputs, hidden = self.rnn(packed, hidden)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        attn_outputs = self.attn_block(outputs, memory)
        return attn_outputs, hidden


class MaskedMeanAggr(nn.Module):

    def __init__(self):
        super(MaskedMeanAggr, self).__init__()
        self.require_mask = True

    def forward(self, x, dim=1, mask=None):
        if x.size(dim) == 1:
            return x
        if mask is None:
            rv = x.mean(dim=dim, keepdim=True)
        else:
            rv = x.sum(dim=dim) / mask.float().sum(dim=-1)
            rv = rv.unsqueeze(dim)
        return rv


class MaskSelectAggr(nn.Module):

    def __init__(self):
        super(MaskSelectAggr, self).__init__()
        self.require_mask = True

    def forward(self, x, dim=1, mask=None):
        if mask is None:
            rv = x[:,-1,:].unsqueeze(1)
        else:
            rv_lens = mask.squeeze(1).sum(dim=1) - 1
            rv = torch.gather(x, 1, rv_lens.view(-1, 1).unsqueeze(2).repeat(1, 1, x.size(-1)))
        return rv


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


class MeanFFAggr(nn.Module):

    def __init__(self, d_input, d_output):
        super(MeanFFAggr, self).__init__()
        self.ff = nn.Linear(d_input, d_output)

    def forward(self, x, dim=1):
        if x.size(dim) == 1:
            rv = x
        else:
            rv = x.mean(dim=dim, keepdim=True)
        return self.ff(rv)


class MaxAggr(nn.Module):

    def __init__(self):
        super(MaxAggr, self).__init__()

    def forward(self, x, dim=1):
        if x.size(dim) == 1:
            return x
        val, _ = x.max(dim=dim, keepdim=True)
        return val

class MeanAggr(nn.Module):

    def __init__(self):
        super(MeanAggr, self).__init__()

    def forward(self, x, dim=1):
        if x.size(dim) == 1:
            return x
        return x.mean(dim=dim, keepdim=True)


class SumAggr(nn.Module):

    def __init__(self):
        super(SumAggr, self).__init__()

    def forward(self, x, dim=1):
        if x.size(dim) == 1:
            return x
        return x.sum(dim=dim, keepdim=True)


class CrxLayer(nn.Module):

    def __init__(self, self_attn_layer, cross_ctx_layer, ff_layer,
                 n_l_self=1, n_r_self=1, n_crx=1, n_l_ff=1, n_r_ff=1):
        super(CrxLayer, self).__init__()
        self.l_self_layers = clones(self_attn_layer, n_l_self)
        self.r_self_layers = clones(self_attn_layer, n_r_self)
        self.cross_ctx_layers = clones(cross_ctx_layer, n_crx)
        self.l_ff_layers = clones(ff_layer, n_l_ff)
        self.r_ff_layers = clones(ff_layer, n_r_ff)

    def forward(self, l, r, l_mask, r_mask):
        for layer in self.l_self_layers:
            l = layer(l, l_mask)
        for layer in self.r_self_layers:
            r = layer(r, r_mask)
        for layer in self.cross_ctx_layers:
            l, r = layer(l, r, l_mask, r_mask)
        for layer in self.l_ff_layers:
            l = layer(l)
        for layer in self.r_ff_layers:
            r = layer(r)
        return l, r


class ParallelCrxPool(nn.Module):

    def __init__(self, crx_layer, l_pool,  r_pool, n_layers=1):
        super(ParallelCrxPool, self).__init__()
        self.crx_layers = clones(crx_layer, n_layers)
        self.l_pool = l_pool
        self.r_pool = r_pool

    def forward(self, l, r, l_mask, r_mask):
        for layer in self.crx_layers:
            l, r = layer(l, r, l_mask, r_mask)
        l = self.l_pool(l)
        r = self.r_pool(r)
        return l, r


class ConcatCrxPool(nn.Module):

    def __init__(self, crx_layer, pool, n_layers=1):
        super(ConcatCrxPool, self).__init__()
        self.crx_layers = clones(crx_layer, n_layers)
        self.pool = pool

    def forward(self, l, r, l_mask, r_mask):
        for layer in self.crx_layers:
            l, r = layer(l, r, l_mask, r_mask)
        rv = torch.cat([l, r], dim=1)
        rv= self.pool(rv)
        return rv


class GLSeqAggrPool(nn.Module):

    def __init__(self, local_aggr, global_aggr, pool, window=10):
        super(GLSeqAggrPool, self).__init__()
        self.local_aggr = local_aggr
        self.global_aggr = global_aggr
        self.pool = pool
        self.window = window

    def forward(self, x, mask=None):
        if self.window <= 1 or x.shape[1] < self.window:
            return x
        rv = []
        curr_i = 0
        g_out = self.global_aggr(x, mask=mask)
        while curr_i < x.shape[1]:
            chunk = x[:,curr_i:curr_i+self.window,:]
            out = self.local_aggr(chunk)
            out = out + g_out
            rv.append(out)
            curr_i += self.window
        rv = torch.cat(rv, dim=1)
        return rv


class SeqConvLayer(nn.Module):

    def __init__(self, d_model_in, d_model_out, seq_len):
        super(SeqConvLayer, self).__init__()
        self.seq_len_threshold = seq_len
        self.conv = nn.Conv1d(d_model_in, d_model_out, kernel_size=seq_len, stride=seq_len)

    def forward(self, x):
        if x.shape[1] < self.seq_len_threshold: return x
        if self.seq_len_threshold <= 1: return x
        rv = self.conv(x.transpose(1,2)).transpose(1,2)
        return rv


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


class FFMaxOutLayer(nn.Module):

    def __init__(self, d_input, d_model, pool_size=2):
        super(FFMaxOutLayer, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.pool_size = pool_size
        self.d_output = self.d_model // self.pool_size
        self.ff = nn.Linear(self.d_input, self.d_model)
        self.max_out = MaxOut(self.pool_size)
        init_weights(self, base_model_type="rnn")

    def forward(self, x):
        x = torch.tanh(self.ff(x))
        x = self.max_out(x)
        return x


class ConditionalMaxPoolLayer(nn.Module):

    def __init__(self, d_model, output_seq_len):
        super(ConditionalMaxPoolLayer, self).__init__()
        self.output_seq_len = output_seq_len
        self.d_model = d_model

    @staticmethod
    def pad_to_seq_len(x, new_seq_len, fill_val=-1e9):
        if x.shape[1] == new_seq_len: return x
        padded = torch.zeros(x.shape[0], new_seq_len, x.shape[2]).type(torch.FloatTensor).to(device())
        padded.fill_(fill_val)
        padded[:, :x.shape[1], :] = x
        return padded

    def forward(self, x):
        if self.output_seq_len < 1 or x.shape[1] <= self.output_seq_len:
            return x
        max_pool_factor = math.ceil(x.shape[1] / self.output_seq_len)
        padded_x = self.pad_to_seq_len(x, max_pool_factor*self.output_seq_len)
        padded_x = padded_x.view(padded_x.shape[0], self.output_seq_len, -1, padded_x.shape[2])
        padded_x, padded_x_mi = padded_x.max(2)
        return padded_x


class GLRP(nn.Module):

    def __init__(self, seq_conv, global_rnn):
        super(GLRP, self).__init__()
        self.seq_conv = seq_conv
        self.global_rnn = global_rnn

    def forward(self, x, mask=None):
        lens = None
        if mask is not None: lens = mask.squeeze(1).sum(dim=1)
        outputs, _ = self.global_rnn(x, lens=lens)
        global_info = outputs[:,-1,:].unsqueeze(1)
        rv = self.seq_conv(x) + global_info
        return rv


class PRP(nn.Module):

    def __init__(self, pool_before, rnn, pool_after):
        super(PRP, self).__init__()
        self.pool_before = pool_before
        self.rnn = rnn
        self.pool_after = pool_after

    def forward(self, x, hidden=None, mask=None):
        original_seq = x.shape[1]
        x = self.pool_before(x)
        if x.shape[1] != original_seq: mask =None
        lens=None
        if mask is not None: lens = mask.squeeze(1).sum(1)
        outputs, hidden = self.rnn(x, hidden, lens=lens)
        out = outputs[:, -1, :].unsqueeze(1)
        out = self.pool_after(out)
        return out


class CollapseLinear(nn.Module):

    def __init__(self, dim_in, bias=False):
        super(CollapseLinear, self).__init__()
        self.bias = bias
        self.ff = nn.Linear(dim_in, 1, bias=bias)
        init_weights(self)

    def forward(self, x):
        x = self.ff(x)
        return x.squeeze(-1)


class GCN(nn.Module):
    """
    A GCN/Contextualized GCN module operated on dependency graphs.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gcn_drop = nn.Dropout(dropout)

        # gcn layer
        self.W = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            output_dim = self.output_dim if layer_idx == self.num_layers-1 else self.hidden_dim
            self.W.append(nn.Linear(input_dim, output_dim))
        init_weights(self, base_model_type="gcn")

    def forward(self, gcn_inputs, adj):
        """
        :param adj: batch_size * num_vertex * num_vertex
        :param gcn_inputs: batch_size * num_vertex * input_dim
        :return: gcn_outputs: list of batch_size * num_vertex * hidden_dim
                 mask: batch_size * num_vertex * 1. In mask, 1 denotes
                     this vertex is PAD vertex, 0 denotes true vertex.
        """
        # use out degree, assume undirected graph
        denom = adj.sum(2).unsqueeze(2) + 1
        adj_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        gcn_outputs = []
        for l in range(self.num_layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.num_layers - 1 else gAxW
            gcn_outputs.append(gcn_inputs)

        return gcn_outputs, adj_mask


class PadTruncateAggrLayer(nn.Module):

    def __init__(self, d_model_in, d_model_out):
        super(PadTruncateAggrLayer, self).__init__()
        self.d_model_in = d_model_in
        self.d_model_out = d_model_out
        self.resize_layer = nn.Linear(d_model_in, d_model_out, bias=False)
        init_weights(self, base_model_type="transformer")

    def forward(self, x):
        input_dim = x.size(-1)
        rv = x
        if input_dim > self.d_model_in:
            rv = x[:, :self.d_model_in]
        elif input_dim < self.d_model_in:
            new_size = list(x.shape)
            new_size[-1] = self.d_model_in
            new_size = torch.Size(new_size)
            rv = torch.zeros(new_size).type(torch.FloatTensor).to(device())
            rv[:, :input_dim] = x
        rv = torch.tanh(self.resize_layer(rv))
        return rv


class HiddenMaxPoolLayer(nn.Module):

    def __init__(self, d_model, max_pool_factor=2):
        super(HiddenMaxPoolLayer, self).__init__()
        self.max_pool_factor = max_pool_factor
        self.d_model = d_model
        self.max_pool = nn.MaxPool1d(self.max_pool_factor, stride=self.max_pool_factor)

    def forward(self, x):
        if self.max_pool_factor <= 1: return x
        x = self.max_pool(x)
        return x


class HiddenAvgPoolLayer(nn.Module):

    def __init__(self, d_model, avg_pool_factor=2):
        super(HiddenAvgPoolLayer, self).__init__()
        self.avg_pool_factor = avg_pool_factor
        self.d_model = d_model
        self.avg_pool = nn.AvgPool1d(self.avg_pool_factor, stride=self.avg_pool_factor)

    def forward(self, x):
        if self.avg_pool_factor <= 1: return x
        x = self.avg_pool(x)
        return x


class SeqMaxPoolLayer(nn.Module):

    def __init__(self, d_model, max_pool_factor=2, min_seq_len=4, fill_val=-1e9):
        super(SeqMaxPoolLayer, self).__init__()
        self.min_seq_len = min_seq_len
        self.max_pool_factor = max_pool_factor
        self.d_model = d_model
        self.max_pool = nn.MaxPool1d(self.max_pool_factor, stride=self.max_pool_factor)
        self.fill_val = fill_val

    def pad_to_max_pool_size(self, x, fill_val=None):
        fill_val = self.fill_val if fill_val is None else fill_val
        if x.shape[1] <= self.min_seq_len or self.max_pool_factor <= 1:
            return x
        if x.shape[1] % self.max_pool_factor != 0:
            new_size = x.shape[1] + (self.max_pool_factor - x.shape[1] % self.max_pool_factor)
            pad_size = new_size - x.shape[1]
            padded = F.pad(x, (0,pad_size,0,0), value=fill_val)
        else:
            padded = x
        return padded

    def forward(self, x):
        padded_x = self.pad_to_max_pool_size(x)
        if x.shape[1] > self.min_seq_len:
            padded_x = padded_x.transpose(1, 2)
            padded_x = self.max_pool(padded_x)
            padded_x = padded_x.transpose(1, 2)
        return padded_x


class SeqAvgPoolLayer(nn.Module):

    def __init__(self, d_model, pool_factor=2, min_seq_len=4):
        super(SeqAvgPoolLayer, self).__init__()
        self.min_seq_len = min_seq_len
        self.pool_factor = pool_factor
        self.d_model = d_model
        self.pool = nn.AvgPool1d(self.pool_factor, stride=self.pool_factor)

    def pad_to_pool_size(self, x, fill_val=None):
        if x.shape[1] <= self.min_seq_len or self.pool_factor <= 1:
            return x
        if x.shape[1] % self.pool_factor != 0:
            new_size = x.shape[1] + (self.pool_factor - x.shape[1] % self.pool_factor)
            padded = torch.zeros(x.shape[0], new_size, x.shape[2]).type(torch.FloatTensor).to(device())
            padded.fill_(fill_val)
            padded[:, :x.shape[1], :] = x
        else:
            padded = x
        return padded

    def forward(self, x):
        padded_x = x
        # padded_x = self.pad_to_pool_size(x)
        if x.shape[1] > self.min_seq_len:
            padded_x = padded_x.transpose(1, 2)
            padded_x = self.pool(padded_x)
            padded_x = padded_x.transpose(1, 2)
        return padded_x


class NormDepthConv(nn.Module):

    def __init__(self, size, conv, dropout=0.0):
        super(NormDepthConv, self).__init__()
        self.size = size
        self.conv = copy.deepcopy(conv)
        self.sublayer = SublayerConnection(self.size, dropout)

    def forward(self, x):
        x = self.sublayer(x, lambda c: self.conv(c))
        return x


class NormSelfAttn(nn.Module):

    def __init__(self, size, attn, dropout=0.0):
        super(NormSelfAttn, self).__init__()
        self.size = size
        self.attn = copy.deepcopy(attn)
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x, mask=None):
        x = self.sublayer(x, lambda q: self.attn(q, q, q, mask))
        return x


class NormAttn(nn.Module):

    def __init__(self, size, attn, dropout=0.0):
        super(NormAttn, self).__init__()
        self.size = size
        self.attn = copy.deepcopy(attn)
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x, y, y_mask=None):
        x = self.sublayer(x, lambda q: self.attn(q, y, y, y_mask))
        return x


class NormAttnFF(nn.Module):

    def __init__(self, size, attn, ff, dropout=0.0):
        super(NormAttnFF, self).__init__()
        self.size = size
        self.attn = copy.deepcopy(attn)
        self.ff = copy.deepcopy(ff)
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, y, z, z_mask=None):
        x = self.sublayers[0](x, lambda q: self.attn(q, y, z, z_mask))
        x = self.sublayers[1](x, self.ff)
        return x


class NormPosFeedForward(nn.Module):

    def __init__(self, size, ff, dropout):
        super(NormPosFeedForward, self).__init__()
        self.size = size
        self.ff = copy.deepcopy(ff)
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x):
        x = self.sublayer(x, self.ff)
        return x


class NormCrossCtx(nn.Module):

    def __init__(self, size, cross_attn, dropout):
        super(NormCrossCtx, self).__init__()
        self.size = size
        self.c2s_cross_attn = copy.deepcopy(cross_attn)
        self.s2c_cross_attn = copy.deepcopy(cross_attn)
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, ctx, src, ctx_mask=None, src_mask=None):
        c2s = self.sublayers[0](ctx, lambda q: self.c2s_cross_attn(q, src, src, src_mask))
        s2c = self.sublayers[1](src, lambda q: self.s2c_cross_attn(q, ctx, ctx, ctx_mask))
        return c2s, s2c


class SCP2SeqSQuADBeam:

    def __init__(self, enc_h, idx_in_batch, beam_width=4, sos_idx=0, gamma=0.0):
        self.idx_in_batch = idx_in_batch
        self.beam_width = beam_width
        self.gamma = gamma
        sos = torch.ones(1,1).fill_(sos_idx).type(torch.LongTensor).to(device())
        self.curr_candidates = [
            (sos, 0.0, [], enc_h)
        ]

    def update(self, next_vals, next_wids, dec_hs):
        assert len(next_wids) == len(self.curr_candidates)
        next_candidates = []
        for i, tup in enumerate(self.curr_candidates):
            score = tup[1]
            indices = [t for t in tup[2]]
            preds = next_wids[i]
            decoder_hidden = dec_hs[i]
            vals = next_vals[i]
            for bi in range(len(preds)):
                wi = preds[bi]
                val = vals[bi]
                div_penalty = 0.0
                if i > 0: div_penalty = self.gamma * (bi+1)
                new_score = score + val - div_penalty
                new_tgt = torch.ones(1,1).type(torch.LongTensor).fill_(wi).to(device())
                indices.append(wi)
                next_candidates.append((new_tgt, new_score, indices, decoder_hidden))
        next_candidates = sorted(next_candidates, key=lambda t: t[1], reverse=True)
        next_candidates = next_candidates[:self.beam_width]
        self.curr_candidates = next_candidates

    def get_curr_tgt(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[0] for tup in self.curr_candidates], dim=0).type(torch.LongTensor).to(device())

    def get_curr_dec_hidden(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[4] for tup in self.curr_candidates], dim=1).type(torch.FloatTensor).to(device())

    def get_curr_candidate_size(self):
        return len(self.curr_candidates)

    def collect_results(self, topk=1):
        rv = [cand[2] for cand in self.curr_candidates]
        return rv[:topk]


class SCP2AttnSQuADBeam:

    def __init__(self, idx_in_batch, beam_width=4, sos_idx=2, eos_idx=3, gamma=0.0):
        self.idx_in_batch = idx_in_batch
        self.beam_width = beam_width
        self.gamma = gamma
        self.eos_idx = eos_idx
        sos = torch.ones(1,1).fill_(sos_idx).type(torch.LongTensor).to(device())
        self.curr_candidates = [
            (sos, 1.0, [sos_idx])
        ]
        self.completed_insts = []
        self.done = False

    def update(self, next_vals, next_wids):
        assert len(next_wids) == len(self.curr_candidates)
        next_candidates = []
        for i, tup in enumerate(self.curr_candidates):
            prev_tgt = tup[0]
            score = tup[1]
            indices = [t for t in tup[2]]
            preds = next_wids[i]
            vals = next_vals[i]
            for bi in range(len(preds)):
                wi = preds[bi]
                val = vals[bi]
                div_penalty = 0.0
                if bi > 0: div_penalty = self.gamma * (bi+1)
                new_score = score + val - div_penalty
                new_tgt = torch.cat([prev_tgt, torch.ones(1,1).type(torch.LongTensor).fill_(wi).to(device())], dim=1)
                indices.append(wi)
                next_candidates.append((new_tgt, new_score, indices))
        next_candidates = sorted(next_candidates, key=lambda t: t[1], reverse=True)
        next_candidates = next_candidates[:self.beam_width]
        self.curr_candidates = next_candidates
        self.done = len(self.curr_candidates) == 0

    def get_curr_tgt(self):
        if len(self.curr_candidates) == 0: return None
        return torch.cat([tup[0] for tup in self.curr_candidates], dim=0).type(torch.LongTensor).to(device())

    def get_curr_candidate_size(self):
        return len(self.curr_candidates)

    def collect_results(self, topk=1):
        rv = [cand[2] for cand in self.curr_candidates]
        return rv[:topk]


# class Transformer(nn.Module):
#
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad_idx=0):
#         super(Transformer, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
#         self.pad_idx = pad_idx
#
#     def forward(self, src, src_mask, tgt, tgt_mask, start_symbol, regressive=False, use_tf=True):
#         src = src.to(device())
#         tgt = tgt.to(device())
#         src_mask = src_mask.to(device())
#         tgt_mask = tgt_mask.to(device())
#         encoder_output = self.encode(src, src_mask)
#         if regressive:
#             return self.regressive_decode(tgt.shape[1], start_symbol, encoder_output, src_mask, truth_tgt=tgt if use_tf else None)
#         else:
#             out = self.tf_decode(encoder_output, tgt, src_mask, tgt_mask)
#             return self.generator(out)
#
#     def encode(self, src, src_mask):
#         src_embedding = self.src_embed(src)
#         encoder_output = self.encoder(src_embedding, src_mask)
#         return encoder_output
#
#     def regressive_decode(self, max_len, start_symbol, memory, src_mask, truth_tgt=None):
#         start_symbol = torch.ones(1, 1).fill_(start_symbol).type(torch.LongTensor).to(device())
#         ys = start_symbol.repeat(memory.shape[0], 1)
#         op = []
#         for i in range(max_len):
#             prob = self.regressive_decode_step(memory, ys, src_mask, make_std_mask(ys, self.pad_idx))
#             op.append(prob.unsqueeze(1))
#             _, next_word = torch.max(prob, dim=1)
#             next_word = next_word.view(-1, 1)
#             if truth_tgt is not None:
#                 next_word = truth_tgt[:, i].view(-1, 1).to(device()) # input right-shifted by 1 when training
#             ys = torch.cat([ys, copy.deepcopy(next_word)], dim=1)
#         ops = torch.cat(op, dim=1)
#         return ops
#
#     def regressive_decode_step(self, memory, tgt, src_mask, tgt_mask):
#         out = self.tf_decode(memory, tgt, src_mask, tgt_mask)
#         prob = self.generator(out[:, -1])
#         return prob
#
#     def tf_decode(self, memory, tgt, src_mask, tgt_mask):
#         memory = memory.to(device())
#         src_mask = src_mask.to(device())
#         tgt = tgt.to(device())
#         tgt_mask = tgt_mask.to(device())
#         tgt = self.tgt_embed(tgt)
#         decoder_output = self.decoder(tgt, memory, src_mask, tgt_mask)
#         return decoder_output
#
#     def predict(self, decoder_out, topk=1):
#         probs = self.generator(decoder_out)
#         val, indices = probs.topk(topk)
#         return probs, val, indices


class RevEmbeddingGenerator(nn.Module):

    def __init__(self, d_input, fwd_embedding_layer, auto_resize=True,
                 multiply=True):
        super(RevEmbeddingGenerator, self).__init__()
        self.d_input = d_input
        self.auto_resize = auto_resize
        self.fwd_embedding_layer = fwd_embedding_layer
        assert hasattr(self.fwd_embedding_layer, "weight")
        self.d_embedding = self.fwd_embedding_layer.weight.shape[1]
        self.resize_layer = None
        self.multiply = multiply
        if self.d_embedding != self.d_input:
            if self.auto_resize:
                self.resize_layer = nn.Linear(self.d_input, self.d_embedding, bias=False)
                init_weights(self.resize_layer, base_model_type="generator")
            else:
                assert False, "Input dim {} not matching embedding dim {}".format(self.d_input, self.d_embedding)

    def forward(self, x):
        x = x.squeeze()
        weight = self.fwd_embedding_layer.weight.transpose(0,1)
        if self.multiply:
            weight = weight * math.sqrt(self.d_embedding)
        if self.resize_layer is not None:
            x = self.resize_layer(x)
        logits = torch.matmul(x, weight)
        probs = F.log_softmax(logits, dim=-1)
        return probs


class RevPTEmbeddingGenerator(nn.Module):

    def __init__(self, d_input, full_embedding_layer, oov_embedding_layer, oov2wi,
                 auto_resize=True, multiply=True):
        super(RevPTEmbeddingGenerator, self).__init__()
        self.d_input = d_input
        self.auto_resize = auto_resize
        self.full_embedding_layer = full_embedding_layer
        self.oov_embedding_layer = oov_embedding_layer
        assert hasattr(self.full_embedding_layer, "weight")
        assert hasattr(self.oov_embedding_layer, "weight")
        self.d_embedding = self.full_embedding_layer.weight.shape[1]
        assert self.d_embedding == self.oov_embedding_layer.weight.shape[1]
        self.resize_layer = None
        self.multiply = multiply
        self.oov2wi_ind_tsr = torch.zeros(self.oov_embedding_layer.weight.shape[0]).long().to(device())
        assert len(oov2wi) == self.oov_embedding_layer.weight.shape[0]
        for oov_i, wi in oov2wi.items():
            self.oov2wi_ind_tsr[oov_i] = wi
        if self.d_embedding != self.d_input:
            if self.auto_resize:
                self.resize_layer = nn.Linear(self.d_input, self.d_embedding, bias=False)
                init_weights(self.resize_layer, base_model_type="generator")
            else:
                assert False, "Input dim {} not matching embedding dim {}".format(self.d_input, self.d_embedding)

    def forward(self, x):
        x = x.squeeze()
        # with torch.no_grad():
        f_weight = self.full_embedding_layer.weight.transpose(0,1)
        if self.multiply:
            f_weight = f_weight * math.sqrt(self.d_embedding)
        oov_weight = self.oov_embedding_layer.weight.transpose(0,1)
        if self.multiply:
            oov_weight = oov_weight * math.sqrt(self.d_embedding)
        weight = f_weight.index_add_(1, self.oov2wi_ind_tsr, oov_weight)
        if self.resize_layer is not None:
            x = self.resize_layer(x)
        logits = torch.matmul(x, weight)
        probs = F.log_softmax(logits, dim=-1)
        return probs


class FishTailGenerator(nn.Module):

    def __init__(self, d_model, d_hidden, vocab_size):
        super(FishTailGenerator, self).__init__()
        self.proj = nn.Linear(d_model, d_hidden)
        self.ff = nn.Linear(d_hidden, vocab_size)
        init_weights(self)

    def forward(self, x):
        logits = self.ff(self.proj(x))
        probs = F.log_softmax(logits, dim=-1)
        return probs


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
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


class MaskedGenerator(nn.Module):

    def __init__(self, d_model, vocab):
        super(MaskedGenerator, self).__init__()
        self.vocab = vocab
        self.proj = nn.Linear(d_model, vocab)
        init_weights(self)

    def forward(self, x, mask):
        logits = self.proj(x)
        if mask is not None:
            if mask.shape[-1] < self.vocab:
                input_dim = mask.shape[-1]
                new_size = list(x.shape)
                new_size[-1] = self.vocab
                new_size = torch.Size(new_size)
                rv = torch.zeros(new_size).type(torch.ByteTensor).to(device())
                rv[:, :input_dim] = mask
                mask = rv
            logits = logits.masked_fill(mask == 0, -1e10)
        probs = F.softmax(logits, dim=-1)
        return probs


# class Encoder(nn.Module):
#     "Core encoder is a stack of n layers"
#     def __init__(self, layer, n):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, n)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)


class PosEncoder(nn.Module):

    def __init__(self, pos_embed, layer, n, dropout_prob=0.0):
        super(PosEncoder, self).__init__()
        self.pos_embed = copy.deepcopy(pos_embed)
        self.layers = clones(layer, n)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, mask=None):
        x = self.pos_embed(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PosDecoder(nn.Module):

    def __init__(self, pos_embed, layer, n, dropout_prob=0.0):
        super(PosDecoder, self).__init__()
        self.pos_embed = copy.deepcopy(pos_embed)
        self.layers = clones(layer, n)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = self.pos_embed(x)
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask=tgt_mask, mem_mask=src_mask)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6, trainable=False):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features), requires_grad=trainable)
        self.b_2 = nn.Parameter(torch.zeros(features), requires_grad=trainable)
        self.dimension = features
        self.trainable = trainable
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    def __repr__(self):
        return "LayerNorm(dimension={}, trainable={})".format(self.dimension, self.trainable)


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_func):
        "Apply residual connection to any sublayer with the same size."
        layer_output = sublayer_func(x)
        residual_rv = x + self.dropout(layer_output)
        return self.norm(residual_rv)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = copy.deepcopy(self_attn)
        self.feed_forward = copy.deepcopy(feed_forward)
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # This is just a list of two residual connections
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda q: self.self_attn(q, q, q, mask)) # 1st residual connection is around the self attention module
        return self.sublayer[1](x, self.feed_forward) # 2nd residual connection is around the feed forward module


# class Decoder(nn.Module):
#     "Generic N layer decoder with masking."
#     def __init__(self, layer, n):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, n)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, x, memory, tgt_mask=None, src_mask=None):
#         for i, layer in enumerate(self.layers):
#             x = layer(x, memory, tgt_mask, src_mask)
#         return self.norm(x)


class ForeignAttnLayer(nn.Module):

    def __init__(self, size, src_attn, feed_forward, dropout):
        super(ForeignAttnLayer, self).__init__()
        self.size = size
        self.src_attn = copy.deepcopy(src_attn)
        self.feed_forward = copy.deepcopy(feed_forward)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, tgt_mask=None, mem_mask=None):
        m = memory
        x = self.sublayer[0](x, lambda q: self.src_attn(q, m, m, mem_mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = copy.deepcopy(self_attn)
        self.src_attn = copy.deepcopy(src_attn)
        self.feed_forward = copy.deepcopy(feed_forward)
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # 3 residual connections in a decoder layer

    def forward(self, x, memory, tgt_mask=None, mem_mask=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda q: self.self_attn(q, q, q, tgt_mask)) # 1st residual is around self attention
        x = self.sublayer[1](x, lambda q: self.src_attn(q, m, m, mem_mask)) # 2nd around encoder attention
        return self.sublayer[2](x, self.feed_forward) # 3rd around feed forward


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # upper triangular matrix
    return torch.from_numpy(subseq_mask) == 0


def batch_subsequent_mask(seq_size, batch_size):
    "Mask out subsequent positions."
    attn_shape = (batch_size, seq_size, seq_size)
    subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # upper triangular matrix
    return torch.from_numpy(subseq_mask) == 0


def get_mh_attention_weights(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # batch_size x n_heads x seq_len x seq_len, i.e. attn score on each word
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)
    p_attn = F.softmax(scores, dim=-1) # batch_size x n_heads x seq_len x seq_len, softmax on last dimension, i.e. 3rd dimension attend on 4th dimension
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # batch_size x n_heads x seq_len x seq_len, i.e. attn score on each word
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)
    p_attn = F.softmax(scores, dim=-1) # batch_size x n_heads x seq_len x seq_len, softmax on last dimension, i.e. 3rd dimension attend on 4th dimension
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # attended output, attention vec


class MultiHeadedAttnFF(nn.Module):

    def __init__(self, h, d_model, pos_ff, dropout=0.0):
        super(MultiHeadedAttnFF, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.q_ff = nn.Linear(d_model, d_model)
        self.k_ff = nn.Linear(d_model, d_model)
        self.v_ff = nn.Linear(d_model, d_model)
        self.last_ff = copy.deepcopy(pos_ff)
        init_weights(self, base_model_type="transformer")

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query = self.q_ff(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.k_ff(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.v_ff(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.last_ff(x)
        return x


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # dimesion of keys should be constrained by model hidden size?
        self.h = h
        # self.linears = clones(nn.Linear(d_model, d_model), 3) # d_model to d_model, attn key dimension downsize is achieved through reshaping
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.q_ff = nn.Linear(d_model, d_model)
        self.k_ff = nn.Linear(d_model, d_model)
        self.v_ff = nn.Linear(d_model, d_model)
        self.concat_ff = nn.Linear(d_model, d_model)
        init_weights(self, base_model_type="transformer")

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # batch_size x seq_len x d_model => batch_size x n_heads x seq_len x d_k
        query = self.q_ff(query).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
        key = self.k_ff(key).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
        value = self.v_ff(value).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
        # query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) # batch_size x n_heads x seq_len x d_k
        #                      for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # batch_size x n_heads x seq_len x d_k => batch_size x n_heads x seq_len x d_k
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # batch_size x n_heads x seq_len x d_k => batch_size x seq_len x d_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.concat_ff(x)
        return x


class ConvMultiHeadedAttention(nn.Module):

    def __init__(self, conv, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(ConvMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h # dimesion of keys should be constrained by model hidden size?
        self.h = h
        self.ffs = clones(nn.Linear(d_model, d_model), 3) # d_model to d_model, attn key dimension downsize is achieved through reshaping
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.last_conv = copy.deepcopy(conv)
        init_weights(self, base_model_type="transformer")

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # batch_size x seq_len x d_model => batch_size x n_heads x seq_len x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # batch_size x n_heads x seq_len x d_k
                             for l, x in zip(self.ffs, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # batch_size x n_heads x seq_len x d_k => batch_size x n_heads x seq_len x d_k
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # batch_size x n_heads x seq_len x d_k => batch_size x seq_len x d_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.last_conv(x)
        return x


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        init_weights(self, base_model_type="transformer")

    def forward(self, x):
        l1_output = F.relu(self.layer_1(x))
        l1_output = self.dropout(l1_output)
        l2_output = self.layer_2(l1_output)
        return l2_output


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, k=7, bias=False, dropout=0.0):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k//2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        self.dropout = nn.Dropout(dropout)
        init_weights(self, base_model_type="transformer")

    def forward(self, x):
        x = x.transpose(1,2)
        return self.dropout(self.pointwise_conv(self.dropout(self.depthwise_conv(x))).transpose(1,2))


class PosDepthSepConv(nn.Module):

    def __init__(self, in_ch, out_ch, hid_ch, k=7, dropout=0.0):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=hid_ch, kernel_size=k, groups=in_ch, padding=k//2)
        self.hidden_conv = nn.Conv1d(in_channels=hid_ch, out_channels=hid_ch, kernel_size=k, groups=hid_ch, padding=k//2)
        self.pointwise_conv = nn.Conv1d(in_channels=hid_ch, out_channels=out_ch, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        init_weights(self, base_model_type="transformer")

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.dropout(self.depthwise_conv(x))
        x = self.dropout(self.hidden_conv(x))
        x = self.dropout(self.pointwise_conv(x))
        return x.transpose(1,2)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).type(torch.FloatTensor).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).type(torch.FloatTensor) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # mark as not learnable parameters, but still part of the state

    def forward(self, x, pe_expand_dim=None):
        encoding_vals = self.pe
        if pe_expand_dim is not None:
            encoding_vals = self.pe.unsqueeze(pe_expand_dim)
        x = x + encoding_vals[:, :x.size(1), :] # just reads the first seq_len positional embedding values
        return x


class TimeStepEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(TimeStepEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).type(torch.FloatTensor).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).type(torch.FloatTensor) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, t):
        x = x + self.pe[:, t]
        return x


class NoamOptimizer:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        # print("noam lr {}".format(self._rate))

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        rv = {}
        rv["_step"] = self._step
        rv["warmup"] = self.warmup
        rv["factor"] = self.factor
        rv["model_size"] = self.model_size
        rv["_rate"] = self._rate
        rv["opt_state_dict"] = self.optimizer.state_dict()
        return rv

    def load_state_dict(self, state_dict):
        self._step = state_dict["_step"]
        self.warmup = state_dict["warmup"]
        self.factor = state_dict["factor"]
        self.model_size = state_dict["model_size"]
        self._rate = state_dict["_rate"]
        self.optimizer.load_state_dict(state_dict["opt_state_dict"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device())


class LabelSmoothing(nn.Module):
    "Label smoothing actually starts to penalize the model if it gets very confident about a given choice"
    def __init__(self, size, padding_idx, smoothing=0.0, reduction="sum"):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        x = x.to(device())
        target = target.to(device())
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        indices = target.data.unsqueeze(1)
        true_dist.scatter_(1, indices, self.confidence)
        if self.padding_idx is not None:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.shape[0] > 0: true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


class ResizePositionWiseFeedForward(nn.Module):

    def __init__(self, d_model_in, d_model_out, d_ff, dropout_prob=0.0):
        super(ResizePositionWiseFeedForward, self).__init__()
        self.layer_1 = nn.Linear(d_model_in, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model_out)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        l1_output = F.relu(self.layer_1(x))
        l1_output = self.dropout(l1_output)
        l2_output = self.layer_2(l1_output)
        return l2_output


class SequentialDecoderLayer(nn.Module):

    def __init__(self, size, attn, pos_ff, dropout, n_attn_units=1):
        super(SequentialDecoderLayer, self).__init__()
        c = copy.deepcopy
        self.size = size
        self.pos_ff = c(pos_ff)
        self.self_attn = c(attn)
        self.p_attns = clones(attn, n_attn_units)
        self.n_attn_units = n_attn_units
        self.sublayers = clones(SublayerConnection(size, dropout), n_attn_units + 2)

    def forward(self, tgt, tgt_mask, attn_in_tup_list):
        assert len(attn_in_tup_list) == self.n_attn_units
        rv = self.sublayers[0](tgt, lambda q: self.self_attn(q, q, q, tgt_mask))
        for i, t in enumerate(attn_in_tup_list):
            rv = self.sublayers[i + 1](rv, lambda q: self.p_attns[i](q, t[0], t[0], t[1]))
        rv = self.sublayers[-1](rv, self.pos_ff)
        return rv


class SequentialDecoder(nn.Module):

    def __init__(self, d_model, layer, n, pos_embed=None):
        super(SequentialDecoder, self).__init__()
        self.d_model = d_model
        self.layers = clones(layer, n)
        self.pos_embed = copy.deepcopy(pos_embed) if pos_embed is not None else None

    def forward(self, tgt, tgt_mask, attn_in_tup_list):
        rv = tgt
        if self.pos_embed is not None:
            rv = self.pos_embed(rv)
        for layer in self.layers:
            rv = layer(rv, tgt_mask, attn_in_tup_list)
        return rv


class LRDecayOptimizer:
    """A simple wrapper class for learning rate scheduling"""
    def __init__(self, optimizer, initial_lr,
                 shrink_factor=0.5,
                 min_lr=0.0001,
                 past_scores_considered=2,
                 verbose=False,
                 score_method="max",
                 max_fail_limit=1):
        self.optimizer = optimizer
        self.curr_lr = initial_lr
        self.shrink_factor = shrink_factor
        self.past_scores_considered = past_scores_considered
        self.verbose = verbose
        self.min_lr = min_lr
        self.past_scores_list = []
        self.score_method = score_method
        self.max_fail_limit = max_fail_limit
        self.curr_fail_count = 0
        self._commit_lr()

    def state_dict(self):
        sd = {
            "opt_sd": self.optimizer.state_dict(),
            "curr_lr": self.curr_lr,
            "shrink_factor": self.shrink_factor,
            "past_scores_considered": self.past_scores_considered,
            "verbose": self.verbose,
            "min_lr": self.min_lr,
            "past_scores_list": self.past_scores_list,
            "score_method": self.score_method,
            "max_fail_limit": self.max_fail_limit,
            "curr_fail_count": self.curr_fail_count,
        }
        return sd

    def load_state_dict(self, state_dict):
        self.curr_lr = state_dict["curr_lr"]
        self.shrink_factor = state_dict["shrink_factor"]
        self.past_scores_considered = state_dict["past_scores_considered"]
        self.verbose = state_dict["verbose"]
        self.min_lr = state_dict["min_lr"]
        self.past_scores_list = state_dict["past_scores_list"]
        self.score_method = state_dict["score_method"]
        self.max_fail_limit = state_dict["max_fail_limit"]
        self.curr_fail_count = state_dict["curr_fail_count"]
        self.optimizer.load_state_dict(state_dict["opt_sd"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device())

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _commit_lr(self):
        self.curr_lr = max(self.min_lr, self.curr_lr)
        self.optimizer.param_groups[0]['lr'] = self.curr_lr

    def _score_func(self, new_score, new_score_method=None):
        score_method = self.score_method if new_score_method is None else new_score_method
        if score_method=="max":
            return new_score > max(self.past_scores_list)
        elif score_method == "min":
            return new_score < min(self.past_scores_list)
        else:
            raise NotImplementedError("Unknown score method " + self.score_method)

    def shrink_learning_rate(self):
        # directly shrink lr
        self.curr_lr *= self.shrink_factor
        self.curr_lr = max(self.curr_lr, self.min_lr)
        if self.verbose: print("lr updated: ", self.curr_lr)
        self._commit_lr()

    def update_learning_rate(self, new_score, new_score_method=None):
        if self.shrink_factor >= 0.999:
            return
        if len(self.past_scores_list) < self.past_scores_considered:
            self.past_scores_list.append(new_score)
            return
        if self._score_func(new_score, new_score_method):
            self.curr_fail_count += 1
            if self.verbose: print("lrd bad_count: ", self.curr_fail_count)
            if self.curr_fail_count >= self.max_fail_limit:
                self.curr_lr *= self.shrink_factor
                self.curr_lr = max(self.curr_lr, self.min_lr)
                if self.verbose: print("lr updated: ", self.curr_lr)
                self._commit_lr()
                self.past_scores_list = [new_score]
                self.curr_fail_count = 0
            else:
                self.past_scores_list.append(new_score)
                if len(self.past_scores_list) > self.past_scores_considered:
                    self.past_scores_list = self.past_scores_list[-self.past_scores_considered:]
        else:
            self.curr_fail_count = 0
            self.past_scores_list.append(new_score)
            if len(self.past_scores_list) > self.past_scores_considered:
                self.past_scores_list = self.past_scores_list[-self.past_scores_considered:]


if __name__ == "__main__":
    print("done")
