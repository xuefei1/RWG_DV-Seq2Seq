import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from components import PositionwiseFeedForward, SublayerConnection
from utils.model_utils import init_weights


class TrainableEmbedding(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=0, multiply_by_sqrt_d_model=False):
        super(TrainableEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        self.multiply_by_sqrt_d_model = multiply_by_sqrt_d_model

    def forward(self, *args):
        x = args[0]
        if torch.isnan(x).any():
            print(x)
            assert False, "NaN detected in indices input"
        rv = self.embedding_layer(x)
        if torch.isnan(rv).any():
            print(torch.isnan(self.embedding_layer.weight).any())
            print(self.embedding_layer.weight)
            assert False, "NaN detected in weights"
        if self.multiply_by_sqrt_d_model:
            rv = rv * math.sqrt(self.d_model)
        return rv


class PreTrainedWordEmbedding(nn.Module):

    def __init__(self, word_mat, d_model, allow_further_training=True, multiply_by_sqrt_d_model=False):
        super(PreTrainedWordEmbedding, self).__init__()
        assert word_mat.shape[1] == d_model
        self.d_model = d_model
        if torch.isnan(word_mat).any():
            print(word_mat)
            assert False, "NaN detected in word_mat"
        self.embedding_layer = nn.Embedding.from_pretrained(word_mat, freeze=not allow_further_training)
        self.multiply_by_sqrt_d_model = multiply_by_sqrt_d_model

    def forward(self, *args):
        x = args[0]
        if torch.isnan(x).any():
            print(x)
            assert False, "NaN detected in indices input"
        rv = self.embedding_layer(x)
        if torch.isnan(rv).any():
            print(torch.isnan(self.embedding_layer.weight).any())
            print(self.embedding_layer.weight)
            assert False, "NaN detected in weights"
        if self.multiply_by_sqrt_d_model:
            rv = rv * math.sqrt(self.d_model)
        return rv


class PartiallyTrainableEmbedding(nn.Module):

    def __init__(self, d_model, word_mat, trainable_vocab_size, padding_idx=0, multiply_by_sqrt_d_model=False):
        super(PartiallyTrainableEmbedding, self).__init__()
        assert word_mat.shape[1] == d_model
        self.d_model = d_model
        self.fixed_embedding_layer = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.trained_embedding_layer = nn.Embedding(trainable_vocab_size, d_model, padding_idx=padding_idx)
        self.multiply_by_sqrt_d_model = multiply_by_sqrt_d_model

    def forward(self, x_fix, x_train):
        assert x_fix.size() == x_train.size(), "x_fix size {} mismatch x_train size {}".format(x_fix.size(), x_train.size())
        x_fix_embedded = self.fixed_embedding_layer(x_fix)
        x_train_embedded = self.trained_embedding_layer(x_train)
        rv = x_fix_embedded + x_train_embedded
        if self.multiply_by_sqrt_d_model:
            rv = rv * math.sqrt(self.d_model)
        return rv


class ResizeWrapperEmbedding(nn.Module):

    def __init__(self, d_model, embed_layer, multiply_by_sqrt_d_model=False):
        super(ResizeWrapperEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model = d_model
        self.resize_layer = nn.Linear(embed_layer.d_model, self.d_model, bias=False)
        self.multiply_by_sqrt_d_model = multiply_by_sqrt_d_model
        init_weights(self.resize_layer)

    def forward(self, *x):
        rv = self.embed_layer(*x)
        rv = self.resize_layer(rv)
        if self.multiply_by_sqrt_d_model:
            rv = rv * math.sqrt(self.d_model)
        return rv


class SortedWrapperEmbedding(nn.Module):

    def __init__(self, d_model, embed_layer):
        super(SortedWrapperEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model = d_model

    def forward(self, x):
        rv = torch.tanh(self.embed_layer(x))
        rv = torch.sort(rv)
        return rv[0]


class SortedWrapperFTEmbedding(nn.Module):

    def __init__(self, d_model_in, d_model_out, embed_layer):
        super(SortedWrapperFTEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model_in = d_model_in
        self.d_model_out = d_model_out
        self.ft_layer = nn.Linear(d_model_in, d_model_out)

    def forward(self, x):
        rv = self.embed_layer(x)
        rv = torch.tanh(self.ft_layer(rv))
        rv = torch.sort(rv)
        return rv[0]


class CBWFineTuneEmbedding(nn.Module):

    def __init__(self, d_model, embed_layer):
        super(CBWFineTuneEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model = d_model
        self.ft_layer = PositionwiseFeedForward(d_model, 4 * d_model)
        self.norm = SublayerConnection(d_model, dropout=0.0)

    def forward(self, x):
        rv = self.embed_layer(x)
        rv = self.norm(rv, self.ft_layer)
        return rv


class CBWCosFineTuneEmbedding(nn.Module):

    def __init__(self, d_model, embed_layer):
        super(CBWCosFineTuneEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.d_model = d_model
        self.ft_layer = PositionwiseFeedForward(d_model, 4 * d_model)
        self.norm = SublayerConnection(d_model, dropout=0.0)

    def forward(self, bow, top_bow):
        rv = self.embed_layer(bow)
        rv = self.norm(rv, self.ft_layer)
        return rv
