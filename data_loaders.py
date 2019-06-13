import os
import pickle
import random
import torch
from tqdm import tqdm
from constants import *
from utils.misc_utils import UniqueDict, label_tsr_to_one_hot_tsr
from utils.lang_utils import *


def loader_reconstruction_test(loader, data_list, col_ind_loader_key_funcs_tups,
                               name="loader", catch_over_recon=True, catch_missing_recon=True):
    """
    # loader_reconstruction_test(train_loader, data_train,
    #                            [(0,
    #                              lambda sgl: "".join([w if w in src_vocab.w2i else OOV_TOKEN for w in sgl]),
    #                              DK_SRC_WID,
    #                              w2i_tsr_to_data,
    #                              [src_vocab.i2w, src_vocab.pad_idx, {}], ""),
    #                             (1,
    #                              lambda sgl: "".join([w if w in tgt_vocab.w2i else OOV_TOKEN for w in sgl]),
    #                              DK_TGT_GEN_WID,
    #                              w2i_tsr_to_data,
    #                              [tgt_vocab.i2w, tgt_vocab.pad_idx, {tgt_vocab.w2i[EOS_TOKEN]}], ""),
    #                             ]
    #                           )
    """
    col_indices = [t[0] for t in col_ind_loader_key_funcs_tups]
    ind_to_data_funcs = [t[1] for t in col_ind_loader_key_funcs_tups]
    loader_keys = [t[2] for t in col_ind_loader_key_funcs_tups]
    key_to_data_funcs = [t[3] for t in col_ind_loader_key_funcs_tups]
    key_to_data_f_args = [t[4] for t in col_ind_loader_key_funcs_tups]
    recon_data_list = [[d[c] for c in col_indices] for d in data_list]
    recon_dicts = [{} for _ in col_indices]
    recon_data_not_in_dict = []
    over_recon_data = []
    recon_failed_data = []
    for row in recon_data_list:
        for i, val in enumerate(row):
            func = ind_to_data_funcs[i]
            d = recon_dicts[i]
            key = func(val)
            if key not in d:
                d[key] = 0
            d[key] += 1
    for batch in loader:
        for i, key in enumerate(loader_keys):
            func = key_to_data_funcs[i]
            recon_d_list = func(batch[key], *key_to_data_f_args[i])
            d = recon_dicts[i]
            for data in recon_d_list:
                if data in d and d[data] > 0:
                    d[data] -= 1
                elif catch_over_recon and data in d and d[data] == 0:
                    over_recon_data.append(data)
                elif catch_missing_recon and data not in d:
                    recon_data_not_in_dict.append(data)
    for d in recon_dicts:
        for k, v in d.items():
            if v != 0:
                recon_failed_data.append(k)
    if len(recon_failed_data) != 0:
        print("Reconstruction from {} failed".format(name))
    else:
        print("Reconstruction from {} success".format(name))
    if catch_missing_recon:
        print("data not found in dict during recon {}".format(len(recon_data_not_in_dict)))
    if catch_over_recon:
        print("data found too many times during recon {}".format(len(over_recon_data)))
    print("{} data failed to clear".format(len(recon_failed_data)))


def word_idx_tsr_to_data(tsr, i2w, pad_idx, other_exclude_inds=set(), word_delim=""):
    rv = []
    for bi in range(tsr.shape[0]):
        tmp = []
        for i in range(tsr.shape[1]):
            wi = tsr[bi, i].item()
            if wi == pad_idx or wi in other_exclude_inds:
                continue
            word = i2w[wi]
            tmp.append(word)
        if isinstance(word_delim, str):
            recon_str = word_delim.join(tmp)
            rv.append(recon_str)
        else:
            rv.append(tmp)
    return rv


class DVSeq2SeqDataLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, wi_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.wi_vocab = wi_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        self.batches = []
        idx = 0
        total_oov_count = 0
        covered_oov_count = 0
        total_tokens_count = 0
        while idx < len(data):
            batch_list = data[idx:min(len(data), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0] + inst[1]
                tgt_seg_list = inst[2]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, max([len(l) for l in src_seg_lists]),
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in tgt_seg_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)

            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()
            total_oov_count += tgt_g_wid.eq(self.tgt_vocab.oov_idx).sum().item()
            covered_oov_count += tgt_c_gate.eq(1).sum().item()
            total_tokens_count += batch_n_tokens.item()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_TGT_SEG_LISTS, tgt_seg_lists),
            ])
            self.batches.append(batch)
            idx += self.batch_size
        percent_oov = total_oov_count/total_tokens_count*100 if total_tokens_count > 0 else 0
        percent_oov_covered = covered_oov_count/total_oov_count*100 if total_oov_count > 0 else 1.0
        print("{}% tokens oov, {}% oov covered".format(percent_oov, percent_oov_covered))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.batches)

    @property
    def n_batches(self):
        return len(self.batches)

    @property
    def src_vocab_size(self):
        return len(self.src_vocab.w2i)

    @property
    def tgt_vocab_size(self):
        return len(self.tgt_vocab.w2i)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.batches)

    def next(self):
        if self.curr_batch_idx >= self.n_batches:
            self.shuffle()
            self.curr_batch_idx = 0
            raise StopIteration()
        next_batch = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1
        return next_batch

    def get_overlapping_data(self, loader):
        if not isinstance(loader, DVSeq2SeqDataLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        for batch in self:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_TGT_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_TGT_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key in my_data:
                    overlapped.append(key)
        return overlapped


class RWGDataLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data, cache_file=None):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        if cache_file is not None and os.path.isfile(cache_file):
            self.batches = pickle.load(open(cache_file, "rb"))
        else:
            self._build_batches(data)
            if cache_file is not None:
                pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)

    def _build_batches(self, data):
        qry_word_insts = data
        qry_word_insts = sorted(qry_word_insts, key=lambda t:(len(t[0]), t[1]), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(qry_word_insts):
            batch_list = qry_word_insts[idx:idx+self.batch_size]
            src_seg_lists = []
            tgt_seg_lists = []
            n_words_list = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                if len(src_seg_list) == 0: continue
                if len(tgt_seg_list) == 0: continue
                n_words = len(tgt_seg_list)
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
                n_words_list.append(n_words)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_tsr_seq_len = max([len(l) for l in tgt_seg_lists])
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, tgt_tsr_seq_len,
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()

            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_WI_N_WORDS, n_words_list),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_TGT_SEG_LISTS, tgt_seg_lists),
            ])
            self.batches.append(batch)
            idx += self.batch_size
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.batches)

    @property
    def n_batches(self):
        return len(self.batches)

    @property
    def src_vocab_size(self):
        return len(self.src_vocab.w2i)

    @property
    def tgt_vocab_size(self):
        return len(self.tgt_vocab.w2i)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.batches)

    def next(self):
        if self.curr_batch_idx >= self.n_batches:
            self.shuffle()
            self.curr_batch_idx = 0
            raise StopIteration()
        next_batch = self.batches[self.curr_batch_idx]
        self.curr_batch_idx += 1
        return next_batch

    def get_overlapping_data(self, loader):
        if not isinstance(loader, RWGDataLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        for batch in self:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_TGT_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_TGT_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key in my_data:
                    overlapped.append(key)
        return overlapped
