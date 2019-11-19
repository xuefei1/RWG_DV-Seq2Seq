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
