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


class FilterLoader:

    def __init__(self, batch_size, qry_vocab, data):
        self.curr_batch_idx = 0
        self.batch_size = batch_size
        self.qry_vocab = qry_vocab
        self.batches = []
        self._build_batches(data)

    def _build_batches(self, insts):
        insts = sorted(insts, key=lambda t:len(t[0]), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            b_qry_seg_lists = []
            b_tgt_list = []
            for inst in batch_list:
                qry_seg_list = inst[0]
                tgt_idx = inst[1]
                b_qry_seg_lists.append(qry_seg_list)
                b_tgt_list.append(tgt_idx)
            qry_vec = gen_word2idx_vec_rep(b_qry_seg_lists, self.qry_vocab.w2i, max([len(l) for l in b_qry_seg_lists]),
                                           pad_idx=self.qry_vocab.pad_idx, oov_idx=self.qry_vocab.oov_idx)
            qry = torch.from_numpy(qry_vec).long()
            tgt = torch.LongTensor(b_tgt_list)
            qry_mask = (qry != self.qry_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)

            batch = UniqueDict([
                (DK_BATCH_SIZE, qry.shape[0]),
                (DK_PAD, self.qry_vocab.pad_idx),
                (DK_QRY_WID, qry),
                (DK_QRY_WID_MASK, qry_mask),
                (DK_TGT_IDX, tgt),
                (DK_B_QRY_SEG_LISTS, b_qry_seg_lists),
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
        return len(self.qry_vocab.w2i)

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
        if not isinstance(loader, FilterLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        for batch in self:
            for i, ctx in enumerate(batch[DK_B_QRY_SEG_LISTS]):
                key = "|".join(ctx)
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_B_QRY_SEG_LISTS]):
                key = "|".join(ctx)
                if key in my_data:
                    overlapped.append(key)
        return overlapped


class CBWLoader:

    def __init__(self, batch_size, vocab, title2query, w2c, pos2idx=None, cbow_window=5,
                 lazy_load=False, exclude_words={}, cache_file=None, max_sup_qry_size=25):
        self.curr_batch_idx = 0
        self.batch_size = batch_size
        self.vocab = vocab
        self.w2c = w2c
        self.exclude_words = exclude_words
        self.max_sup_qry_size = max_sup_qry_size
        self.cbow_window = cbow_window
        self.lazy_load = lazy_load
        self.pos2idx = pos2idx
        self.batches = []
        self.extracted_insts = []
        if cache_file is not None and os.path.isfile(cache_file):
            if self.lazy_load:
                self.extracted_insts = pickle.load(open(cache_file, "rb"))
            else:
                self.batches = pickle.load(open(cache_file, "rb"))
        else:
            if cache_file is not None:
                if self.lazy_load:
                    self.extracted_insts = self._extract_insts_from_t2q(title2query)
                    pickle.dump(self.extracted_insts, open(cache_file, "wb"), protocol=4)
                else:
                    insts = self._extract_insts_from_t2q(title2query)
                    self.batches = self._build_batches(insts)
                    if cache_file is not None:
                        pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)
            else:
                insts = self._extract_insts_from_t2q(title2query)
                self.batches = self._build_batches(insts)
        self.shuffle()

    def _extract_insts_from_t2q(self, title2query):
        def most_freq_pos_tag(pos_tags_list):
            d = {}
            for p in pos_tags_list:
                if p not in d: d[p] = 0
                d[p] += 1
            r = sorted([(k,v) for k,v in d.items()],key=lambda t:t[1],reverse=True)[0][0]
            return r
        def order_by_mowe_sim(tgt_data, tgt_key_func, cand_data_list, cand_key_func, return_scores=False):
            scores = []
            try:
                mowe_tgt = mean_of_w2v(tgt_key_func(tgt_data), self.vocab.w2v)
                for cand in cand_data_list:
                    cand_seg_list = cand_key_func(cand)
                    if len(cand_seg_list) == 0: continue
                    mowe_cand = mean_of_w2v(cand_seg_list, self.vocab.w2v)
                    sim = cosine_sim_np(mowe_cand, mowe_tgt)
                    scores.append((sim, cand))
                scores = sorted(scores, key=lambda t:t[0], reverse=True)
                return [t if return_scores else t[1] for t in scores]
            except ValueError:
                return []
        rv = []
        for title, qd in title2query.items():
            if len(qd) <= 1: continue
            title_words = [w for w in title.seg_list if w not in self.exclude_words]
            if len(title_words) == 0: continue
            for qry_tgt, _ in qd.items():
                for w_tgt in qry_tgt.seg_list:
                    qry_cand_bow_map = []
                    dup_words = {}
                    for qry_cmp, _ in qd.items():
                        if len(qry_cand_bow_map) > self.max_sup_qry_size: break
                        if qry_tgt == qry_cmp: continue
                        for i, w_cand in enumerate(qry_cmp.seg_list):
                            if w_cand in dup_words: continue
                            if w_cand in self.exclude_words or w_cand not in self.vocab.w2v: continue
                            if w_tgt == w_cand: continue
                            qry_cand_bow_map.append( (w_cand, i, qry_cmp) )
                            dup_words[w_cand] = 1
                    # find words to fill the window
                    word_order = order_by_mowe_sim(w_tgt, lambda w:w, qry_cand_bow_map, lambda d: d[0])
                    if len(word_order) == 0: continue
                    word_order = word_order[:self.cbow_window]
                    bow_feats = []
                    for selected in word_order:
                        word, word_idx_in_qry, qry = selected
                        pos_tag = qry.pos_list[word_idx_in_qry]
                        rel_position = (word_idx_in_qry + 1) / len(qry.seg_list)
                        if SD_DK_TW_LIST in qry.extras:
                            tw = qry.extras[SD_DK_TW_LIST][word_idx_in_qry]
                        else:
                            tw = 1.0
                        bow_feats.append((word, pos_tag, rel_position, tw))
                    if len(bow_feats) > 0:
                        rv.append([bow_feats, title_words, w_tgt])
        print("{} BoW instances created".format(len(rv)))
        return rv

    def _build_batches(self, insts):
        rv_batches = []
        idx = 0
        insts = sorted(insts, key=lambda t: (len(t[0]), len(t[1])), reverse=True)
        # num_batches = int(len(insts) / self.batch_size)
        while idx < len(insts):
            batch_list = insts[idx:idx+self.batch_size]
            b_bow_seg_lists = []
            b_pos_tag_seg_lists = []
            b_rel_pos_seg_lists = []
            b_tws_seg_lists = []
            b_title_seg_lists = []
            b_tgt_word_lists = []
            for inst in batch_list:
                word_feat_tup_list = inst[0]
                word_seg_list = [tup[0] for tup in word_feat_tup_list]
                pos_tag_seg_list = [tup[1] for tup in word_feat_tup_list]
                rel_pos_seg_list = [tup[2] for tup in word_feat_tup_list]
                tw_seg_list = [tup[3] for tup in word_feat_tup_list]

                title_seg_list = inst[1]
                tgt_word = inst[2]
                b_bow_seg_lists.append(word_seg_list)
                b_pos_tag_seg_lists.append(pos_tag_seg_list)
                b_rel_pos_seg_lists.append(rel_pos_seg_list)
                b_tws_seg_lists.append(tw_seg_list)
                b_title_seg_lists.append(title_seg_list)
                b_tgt_word_lists.append([tgt_word])
            bow_vec = gen_word2idx_vec_rep(b_bow_seg_lists, self.vocab.w2i, max([len(l) for l in b_bow_seg_lists]),
                                           pad_idx=self.vocab.pad_idx, oov_idx=self.vocab.oov_idx)
            title_vec = gen_word2idx_vec_rep(b_title_seg_lists, self.vocab.w2i, max([len(l) for l in b_title_seg_lists]),
                                             pad_idx=self.vocab.pad_idx, oov_idx=self.vocab.oov_idx)
            tgt_vec = gen_word2idx_vec_rep(b_tgt_word_lists, self.vocab.w2i, max([len(l) for l in b_tgt_word_lists]),
                                           pad_idx=self.vocab.pad_idx, oov_idx=self.vocab.oov_idx)
            bow = torch.from_numpy(bow_vec).long()
            title = torch.from_numpy(title_vec).long()
            tgt = torch.from_numpy(tgt_vec).squeeze(1).long()
            bow_mask = (bow != self.vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            title_mask = (title != self.vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch = UniqueDict([
                (DK_BATCH_SIZE, bow.shape[0]),
                (DK_PAD, self.vocab.pad_idx),
                (DK_BOW_WID, bow),
                (DK_BOW_WID_MASK, bow_mask),
                (DK_TITLE_WID, title),
                (DK_TITLE_WID_MASK, title_mask),
                (DK_TGT_WID, tgt),
                (DK_TGT_N_TOKENS, tgt.shape[0]),
                (DK_B_BOW_SEG_LISTS, b_bow_seg_lists),
                (DK_B_TITLE_SEG_LISTS, b_title_seg_lists),
                (DK_B_TGT_SEG_LISTS, b_tgt_word_lists),
            ])
            rv_batches.append(batch)
            idx += self.batch_size
        return rv_batches

    def shuffle(self):
        random.shuffle(self.batches) if not self.lazy_load else random.shuffle(self.extracted_insts)

    @property
    def n_batches(self):
        if self.lazy_load:
            assert False, "n_batches() on loader not correct when lazy_load is set to True"
        return len(self.batches)

    @property
    def src_vocab_size(self):
        return len(self.vocab.w2i)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        if self.lazy_load:
            assert False, "len() on loader not correct when lazy_load is set to True"
        return len(self.batches)

    def next(self):
        if self.lazy_load:
            if self.curr_batch_idx >= len(self.extracted_insts):
                self.shuffle()
                self.curr_batch_idx = 0
                raise StopIteration()
            next_insts = self.extracted_insts[self.curr_batch_idx:self.curr_batch_idx+self.batch_size]
            next_batch = self._build_batches(next_insts)
            assert len(next_batch) == 1
            next_batch = next_batch[0]
            self.curr_batch_idx += self.batch_size
            return next_batch
        else:
            if self.curr_batch_idx >= self.n_batches:
                self.shuffle()
                self.curr_batch_idx = 0
                raise StopIteration()
            next_batch = self.batches[self.curr_batch_idx]
            self.curr_batch_idx += 1
            return next_batch

    def get_overlapping_data(self, loader):
        if not isinstance(loader, CBWLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        for batch in self:
            for i, ctx in enumerate(batch[DK_B_BOW_SEG_LISTS]):
                key = "|".join(ctx) + "_" + "|".join(batch[DK_B_TITLE_SEG_LISTS][i]) + "_" + "|".join(batch[DK_B_TGT_SEG_LISTS][i])
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_B_BOW_SEG_LISTS]):
                key = "|".join(ctx) + "_" + "|".join(batch[DK_B_TITLE_SEG_LISTS][i]) + "_" + "|".join(batch[DK_B_TGT_SEG_LISTS][i])
                if key in my_data:
                    overlapped.append(key)
        return overlapped


class QRWPairsLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        total_oov_count = 0
        covered_oov_count = 0
        total_tokens_count = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in tgt_seg_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
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
                (DK_SRC_OOV_WID, src_oov),
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
        if not isinstance(loader, QRWPairsLoader):
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


class ESGLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, max([len(l) for l in src_seg_lists]),
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in tgt_seg_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            tgt_w_embed = gen_word2vec_np(tgt_seg_lists, self.tgt_vocab.w2v, self.tgt_vocab.embedding_dim,
                                          max([len(l) for l in tgt_seg_lists]), oov_token=OOV_TOKEN)
            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            tgt_w_embed = torch.from_numpy(tgt_w_embed).float()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_W_EMBED, tgt_w_embed),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
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
        if not isinstance(loader, ESGLoader):
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


class KEGLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data, grid):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.grid = grid
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            b_src_seg_lists = []
            b_tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                b_src_seg_lists.append(src_seg_list)
                b_tgt_seg_lists.append(tgt_seg_list)
            src_vec = gen_word2idx_vec_rep(b_src_seg_lists, self.src_vocab.w2i, max([len(l) for l in b_src_seg_lists]),
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            tgt_vec = gen_word2idx_vec_rep(b_tgt_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in b_tgt_seg_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            tgt_ke_vec = self.grid.get_e_np(b_tgt_seg_lists, max([len(l) for l in b_tgt_seg_lists]), oov_idx=self.tgt_vocab.oov_idx)
            cpy_wids, cpy_gates = gen_cpy_np(b_src_seg_lists, b_tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            tgt_ke = torch.from_numpy(tgt_ke_vec).long()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()

            tgt_arr = tgt_g_wid.view(-1, 1).squeeze(1)
            tgt_ke_arr = tgt_ke.contiguous().view(-1, tgt_ke.size(-1))
            tgt_non_pad_ind = (tgt_arr != self.tgt_vocab.pad_idx).nonzero().squeeze()
            tgt_non_pad_ke = tgt_ke_arr.index_select(dim=0, index=tgt_non_pad_ind)
            tgt_non_pad_ke = tgt_non_pad_ke.contiguous().view(-1, 1).squeeze(1)
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_WKE, tgt_ke),
                (DK_TGT_NON_PAD_WKE, tgt_non_pad_ke),
                (DK_TGT_NON_PAD_IND, tgt_non_pad_ind),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_TGT_E_DIM, self.grid.encoding_dim),
                (DK_SRC_SEG_LISTS, b_src_seg_lists),
                (DK_TGT_SEG_LISTS, b_tgt_seg_lists),
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
        if not isinstance(loader, KEGLoader):
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


class WILoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_tsr_seq_len = max([len(l) for l in tgt_seg_lists])
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, tgt_tsr_seq_len,
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            tgt_oov_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.oov_w2i, tgt_tsr_seq_len,
                                               pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.pad_idx)
            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_oov_wid = torch.from_numpy(tgt_oov_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_OOV_WID, tgt_oov_wid),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
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
        if not isinstance(loader, WILoader):
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


class WIQRWLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, wi_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.wi_vocab = wi_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[2]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        total_oov_count = 0
        covered_oov_count = 0
        total_tokens_count = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[2]
                tgt_seg_list = inst[1]
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
        if not isinstance(loader, WIQRWLoader):
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


class WIA2GBCELoader:

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
        if not isinstance(loader, WIA2GBCELoader):
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


class WIAttn2GenLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data, cache_file=None, group_target_words=False):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self.group_target_words = group_target_words
        if cache_file is not None and os.path.isfile(cache_file):
            self.batches = pickle.load(open(cache_file, "rb"))
        else:
            self._build_batches(data)
            if cache_file is not None:
                pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)

    @staticmethod
    def _expand_data(data):
        qry_word_insts = []
        uniques = set()
        for inst in tqdm(data, desc="Expanding data", ascii=True):
            src_seg_list = inst[0]
            # src_seg_list = inst[2] # keywords
            tgt_seg_list = inst[1]
            if len(src_seg_list) == 0: continue
            if len(tgt_seg_list) <= 1: continue
            for word in tgt_seg_list:
                key = "|".join([" ".join(src_seg_list), word])
                if key not in uniques:
                    uniques.add(key)
                    qry_word_insts.append([src_seg_list, [word], len(tgt_seg_list)])
        print("{} qry word instances created".format(len(qry_word_insts)))
        return qry_word_insts

    def _build_batches(self, data):
        qry_word_insts = data
        if not self.group_target_words:
            qry_word_insts = self._expand_data(qry_word_insts)
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
                n_words = inst[2] if len(inst) == 3 else len(tgt_seg_list)
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
            # tgt_oov_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.oov_w2i, tgt_tsr_seq_len,
            #                                    pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.pad_idx)
            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            # tgt_oov_wid = torch.from_numpy(tgt_oov_vec).long()
            # tgt_c_wid = torch.from_numpy(cpy_wids).long()
            # tgt_c_gate = torch.from_numpy(cpy_gates).float()
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
                # (DK_TGT_OOV_WID, tgt_oov_wid),
                (DK_TGT_GEN_WID, tgt_g_wid),
                # (DK_TGT_GEN_OOV_WID, tgt_oov_wid),
                # (DK_TGT_CPY_WID, tgt_c_wid),
                # (DK_TGT_CPY_GATE, tgt_c_gate),
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
        if not isinstance(loader, WIAttn2GenLoader):
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


class WCILoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data, cache_file=None, expand_data=True):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self.expand_data = expand_data
        if cache_file is not None and os.path.isfile(cache_file):
            self.batches = pickle.load(open(cache_file, "rb"))
        else:
            self._build_batches(data)
            if cache_file is not None:
                pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)

    @staticmethod
    def _expand_data(data):
        qry_word_insts = []
        uniques = set()
        for inst in tqdm(data, desc="Expanding data", ascii=True):
            src_seg_list = inst[0]
            tgt_seg_list = inst[1]
            if len(tgt_seg_list) <= 1: continue
            for word in tgt_seg_list:
                key = "|".join([" ".join(src_seg_list), word])
                if key not in uniques:
                    uniques.add(key)
                    qry_word_insts.append([src_seg_list, list(word), len(tgt_seg_list), tgt_seg_list])
        print("{} qry word instances created".format(len(qry_word_insts)))
        return qry_word_insts

    def _build_batches(self, data):
        if self.expand_data:
            data = self._expand_data(data)
        qry_word_insts = sorted(data, key=lambda t:(len(t[0]), t[1]), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(qry_word_insts):
            batch_list = qry_word_insts[idx:idx+self.batch_size]
            src_seg_lists = []
            tgt_seg_lists = []
            all_tgt_words_seg_lists = []
            # n_words_list = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1] + [EOS_TOKEN]
                # n_words = inst[2]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
                # n_words_list.append(n_words)
                if self.expand_data:
                    all_tgt_words_seg_lists.append(inst[3])
                else:
                    all_tgt_words_seg_lists.append(tgt_seg_list)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_tsr_seq_len = max([len(l) for l in tgt_seg_lists])
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, tgt_tsr_seq_len,
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                # (DK_WI_N_WORDS, n_words_list),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_TGT_SEG_LISTS, tgt_seg_lists),
                (DK_ALL_TGT_WORDS_SEG_LISTS, all_tgt_words_seg_lists),
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
        if not isinstance(loader, WCILoader):
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


class WIAttn2KEGLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data, grid, cache_file=None):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.grid = grid
        self.batches = []
        self.curr_batch_idx = 0
        if cache_file is not None and os.path.isfile(cache_file):
            self.batches = pickle.load(open(cache_file, "rb"))
        else:
            self._build_batches(data)
            if cache_file is not None:
                pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)

    @staticmethod
    def _expand_data(data):
        qry_word_insts = []
        for inst in tqdm(data, desc="Expanding data", ascii=True):
            src_seg_list = inst[0]
            # src_seg_list = inst[2]
            tgt_seg_list = inst[1]
            if len(tgt_seg_list) <= 1: continue
            for word in tgt_seg_list:
                qry_word_insts.append([src_seg_list, [word], len(tgt_seg_list)])
        print("{} qry word instances created".format(len(qry_word_insts)))
        return qry_word_insts

    def _build_batches(self, data):
        qry_word_insts = self._expand_data(data)
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
                n_words = inst[2]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
                n_words_list.append(n_words)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in tgt_seg_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            tgt_ke_vec = self.grid.get_e_np(tgt_seg_lists, max([len(l) for l in tgt_seg_lists]),
                                            oov_idx=self.tgt_vocab.oov_idx)
            cpy_wids, cpy_gates = gen_cpy_np(src_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            tgt_ke = torch.from_numpy(tgt_ke_vec).long()
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
                (DK_TGT_WKE, tgt_ke),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
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
        if not isinstance(loader, WIAttn2GenLoader):
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


class WIBinClassifyLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, label_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.label_vocab = label_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            word_seg_lists = []
            label_list = []
            for inst in batch_list:
                src_seg_list = inst[0]
                cand_words = inst[1]
                labels = [lab for lab in inst[2]]
                assert len(cand_words) == len(labels)
                src_seg_lists.append(src_seg_list)
                word_seg_lists.append(cand_words)
                label_list.append(labels)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            cdw_tsr_seq_len = max([len(l) for l in word_seg_lists])
            cdw_vec = gen_word2idx_vec_rep(word_seg_lists, self.tgt_vocab.w2i, cdw_tsr_seq_len,
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            cdw_oov_vec = gen_word2idx_vec_rep(word_seg_lists, self.tgt_vocab.oov_w2i, cdw_tsr_seq_len,
                                               pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.pad_idx)
            labels_vec = gen_word2idx_vec_rep(label_list, self.label_vocab.w2i, cdw_tsr_seq_len,
                                              pad_idx=self.label_vocab.pad_idx, oov_idx=self.label_vocab.oov_idx)
            tgt = torch.from_numpy(labels_vec).long()
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            cdw = torch.from_numpy(cdw_vec).long()
            cdw_oov = torch.from_numpy(cdw_oov_vec).long()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (cdw != self.tgt_vocab.pad_idx).data.sum()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_QRY_WID, src),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_CAND_WORD_WID, cdw),
                (DK_CAND_WORD_OOV_WID, cdw_oov),
                (DK_TGT, tgt),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_CAND_WORD_SEG_LISTS, word_seg_lists),
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
        if not isinstance(loader, WIBinClassifyLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        for batch in self:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_CAND_WORD_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_CAND_WORD_SEG_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key in my_data:
                    overlapped.append(key)
        return overlapped


class TMCLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, data):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batches = []
        self.curr_batch_idx = 0
        self._build_batches(data)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:len(t[0]), reverse=True)
        self.batches = []
        idx = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_val_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_val_list = inst[1]
                if len(src_seg_list) != len(tgt_val_list):
                    print("Bad data pair {} | {} ignored".format(" ".join(src_seg_list), " ".join(tgt_val_list)))
                    continue
                src_seg_lists.append(src_seg_list)
                tgt_val_lists.append(tgt_val_list)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_val_lists, self.tgt_vocab.w2i, max([len(l) for l in tgt_val_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            tgt_val = torch.from_numpy(tgt_vec).long()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            batch_n_tokens = (tgt_val != self.tgt_vocab.pad_idx).data.sum()
            assert src.size() == src_oov.size() == tgt_val.size()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_QRY_WID, src),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_VAL, tgt_val),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_TGT_VAL_LISTS, tgt_val_lists),
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
        if not isinstance(loader, TMCLoader):
            print("type mismatch, no overlaps by default")
            return []
        overlapped = []
        my_data = {}
        for batch in self:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_TGT_VAL_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key not in my_data:
                    my_data[key] = 0
                my_data[key] += 1
        for batch in loader:
            for i, ctx in enumerate(batch[DK_SRC_SEG_LISTS]):
                rsp = batch[DK_TGT_VAL_LISTS][i]
                key = "|".join([" ".join(ctx), " ".join(rsp)])
                if key in my_data:
                    overlapped.append(key)
        return overlapped


class DVQRWLoader:

    def __init__(self, batch_size, src_vocab, tgt_vocab, wi_vocab, data, dv_selector,
                 dv_topk=100, cache_file=None):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.wi_vocab = wi_vocab
        self.dv_selector = dv_selector
        self.dv_topk = dv_topk
        self.batches = []
        self.curr_batch_idx = 0
        if cache_file is not None and os.path.isfile(cache_file):
            self.batches = pickle.load(open(cache_file, "rb"))
        else:
            self._build_batches(data)
            if cache_file is not None:
                pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1]+t[2]), len(t[2]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        total_oov_count = 0
        covered_oov_count = 0
        total_tokens_count = 0
        report_bi = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            wi_tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                wi_tgt_seg_list = inst[2]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
                wi_tgt_seg_lists.append(wi_tgt_seg_list)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.src_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.pad_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_vocab.w2i, max([len(l) for l in tgt_seg_lists]),
                                           pad_idx=self.tgt_vocab.pad_idx, oov_idx=self.tgt_vocab.oov_idx)
            # wi_vec = gen_word2idx_vec_rep(wi_tgt_seg_lists, self.wi_vocab.w2i, max([len(l) for l in wi_tgt_seg_lists]),
            #                               pad_idx=self.src_vocab.pad_idx, oov_idx=self.src_vocab.oov_idx)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            src_mask = (src != self.src_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            # wi = torch.from_numpy(wi_vec).long()

            wi_batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_SRC_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_SRC_WID_MASK, src_mask),
            ])
            wi_tsr, mask, wi_words = self.dv_selector(wi_batch, new_topk=self.dv_topk)
            wi_tgt_seg_lists = word_idx_tsr_to_data(wi_tsr.squeeze(1), self.wi_vocab.i2w, self.wi_vocab.pad_idx, word_delim=None)
            cpy_wids, cpy_gates = gen_cpy_np(wi_tgt_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_vocab.w2i)
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            batch_n_tokens = (tgt_g_wid != self.tgt_vocab.pad_idx).data.sum()
            total_oov_count += tgt_g_wid.eq(self.tgt_vocab.oov_idx).sum().item()
            covered_oov_count += tgt_c_gate.eq(1).sum().item()
            total_tokens_count += batch_n_tokens.item()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_SV_WID, tgt_g_wid),
                (DK_TGT_WI_WID, wi_tsr),
                (DK_TGT_DV_WID, tgt_c_wid),
                (DK_TGT_DV_GATE, tgt_c_gate),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_SRC_ORI_SEG_LISTS, src_seg_lists),
                (DK_WI_SEG_LISTS, wi_tgt_seg_lists),
                (DK_TGT_SEG_LISTS, tgt_seg_lists),
                (DK_WI_GEN_SEQ_LEN, wi_tsr.shape[1])
            ])
            self.batches.append(batch)
            idx += self.batch_size
            report_bi += 1
            if report_bi >= 10:
                print("generated {} batches of {}".format(len(self.batches), int(len(insts) / self.batch_size)))
                report_bi = 0
        percent_oov = total_oov_count/total_tokens_count*100 if total_tokens_count > 0 else 0
        percent_oov_covered = covered_oov_count/total_oov_count*100 if total_oov_count > 0 else 1.0
        print("{}% tokens oov in tgt, {}% oov covered".format(percent_oov, percent_oov_covered))
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
        if not isinstance(loader, DVQRWLoader):
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


class PreGenWIQRWLoader:

    def __init__(self, batch_size, uniform_vocab, tgt_sv_vocab, data, cache_file=None):
        self.batch_size = batch_size
        self.tgt_sv_vocab = tgt_sv_vocab
        self.uniform_vocab = uniform_vocab
        self.batches = []
        self.curr_batch_idx = 0
        if cache_file and os.path.isfile(cache_file):
            self.batches = pickle.load(open(cache_file, "rb"))
        else:
            self._build_batches(data)
            if cache_file is not None:
                pickle.dump(self.batches, open(cache_file, "wb"), protocol=4)

    def _build_batches(self, data):
        insts = sorted(data, key=lambda t:(len(t[0]), len(t[1]+t[2]), len(t[2]), len(t[1])), reverse=True)
        self.batches = []
        idx = 0
        total_oov_count = 0
        covered_oov_count = 0
        total_tokens_count = 0
        while idx < len(insts):
            batch_list = insts[idx:min(len(insts), idx+self.batch_size)]
            src_seg_lists = []
            tgt_seg_lists = []
            wi_tgt_seg_lists = []
            for inst in batch_list:
                src_seg_list = inst[0]
                tgt_seg_list = inst[1]
                wi_tgt_seg_list = inst[2]
                tgt_seg_list = tgt_seg_list + [EOS_TOKEN]
                src_seg_lists.append(src_seg_list)
                tgt_seg_lists.append(tgt_seg_list)
                wi_tgt_seg_lists.append(wi_tgt_seg_list)
            src_tsr_seq_len = max([len(l) for l in src_seg_lists])
            src_vec = gen_word2idx_vec_rep(src_seg_lists, self.uniform_vocab.w2i, src_tsr_seq_len,
                                           pad_idx=self.uniform_vocab.pad_idx, oov_idx=self.uniform_vocab.oov_idx)
            src_oov_vec = gen_word2idx_vec_rep(src_seg_lists, self.uniform_vocab.oov_w2i, src_tsr_seq_len,
                                               pad_idx=self.uniform_vocab.pad_idx, oov_idx=self.uniform_vocab.pad_idx)
            tgt_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.tgt_sv_vocab.w2i, max([len(l) for l in tgt_seg_lists]),
                                           pad_idx=self.tgt_sv_vocab.pad_idx, oov_idx=self.tgt_sv_vocab.oov_idx)
            tgt_input_vec = gen_word2idx_vec_rep(tgt_seg_lists, self.uniform_vocab.w2i,
                                                 max([len(l) for l in tgt_seg_lists]),
                                                 pad_idx=self.uniform_vocab.pad_idx, oov_idx=self.uniform_vocab.oov_idx)
            wi_vec = gen_word2idx_vec_rep(wi_tgt_seg_lists, self.uniform_vocab.w2i, max([len(l) for l in wi_tgt_seg_lists]),
                                          pad_idx=self.uniform_vocab.pad_idx, oov_idx=self.uniform_vocab.oov_idx)
            src = torch.from_numpy(src_vec).long()
            src_oov = torch.from_numpy(src_oov_vec).long()
            src_mask = (src != self.uniform_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)
            wi_tsr = torch.from_numpy(wi_vec).long()
            wi_tsr_mask = (wi_tsr != self.uniform_vocab.pad_idx).type(torch.ByteTensor).unsqueeze(1)

            cpy_wids, cpy_gates = gen_cpy_np(wi_tgt_seg_lists, tgt_seg_lists, tgt_vec.shape[1], self.tgt_sv_vocab.w2i)
            tgt_g_wid = torch.from_numpy(tgt_vec).long()
            tgt_c_wid = torch.from_numpy(cpy_wids).long()
            tgt_c_gate = torch.from_numpy(cpy_gates).float()
            tgt_input_wid = torch.from_numpy(tgt_input_vec).long()
            batch_n_tokens = (tgt_g_wid != self.tgt_sv_vocab.pad_idx).data.sum()
            total_oov_count += tgt_g_wid.eq(self.tgt_sv_vocab.oov_idx).sum().item()
            covered_oov_count += tgt_c_gate.eq(1).sum().item()
            total_tokens_count += batch_n_tokens.item()
            batch = UniqueDict([
                (DK_BATCH_SIZE, src.shape[0]),
                (DK_PAD, self.tgt_sv_vocab.pad_idx),
                (DK_SRC_WID, src),
                (DK_QRY_WID, src),
                (DK_SRC_OOV_WID, src_oov),
                (DK_SRC_WID_MASK, src_mask),
                (DK_QRY_WID_MASK, src_mask),
                (DK_TGT_WID, tgt_g_wid),
                (DK_TGT_INPUT_WID, tgt_input_wid),
                (DK_TGT_SV_WID, tgt_g_wid),
                (DK_TGT_WI_WID, wi_tsr),
                (DK_TGT_WI_WID_MASK, wi_tsr_mask),
                (DK_TGT_DV_WID, tgt_c_wid),
                (DK_TGT_DV_GATE, tgt_c_gate),
                (DK_TGT_GEN_WID, tgt_g_wid),
                (DK_TGT_CPY_WID, tgt_c_wid),
                (DK_TGT_CPY_GATE, tgt_c_gate),
                (DK_TGT_N_TOKENS, batch_n_tokens),
                (DK_SRC_SEG_LISTS, src_seg_lists),
                (DK_SRC_ORI_SEG_LISTS, src_seg_lists),
                (DK_WI_SEG_LISTS, wi_tgt_seg_lists),
                (DK_TGT_SEG_LISTS, tgt_seg_lists),
                (DK_WI_GEN_SEQ_LEN, wi_tsr.shape[1])
            ])
            self.batches.append(batch)
            idx += self.batch_size
        percent_oov = total_oov_count/total_tokens_count*100 if total_tokens_count > 0 else 0
        percent_oov_covered = covered_oov_count/total_oov_count*100 if total_oov_count > 0 else 1.0
        print("{}% tokens oov in tgt, {}% oov covered".format(percent_oov, percent_oov_covered))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.batches)

    @property
    def n_batches(self):
        return len(self.batches)

    @property
    def src_vocab_size(self):
        return len(self.uniform_vocab.w2i)

    @property
    def tgt_vocab_size(self):
        return len(self.tgt_sv_vocab.w2i)

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
        if not isinstance(loader, PreGenWIQRWLoader):
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
