import numpy as np


def gen_tag_np(tag_seg_lists, max_len, tag2idx, pad_idx=0, oov_idx=1):
    rv = np.zeros((len(tag_seg_lists), max_len))
    rv.fill(pad_idx)
    for i, tag_seg_list in enumerate(tag_seg_lists):
        for j, tag in enumerate(tag_seg_list):
            if tag in tag2idx:
                rv[i,j] = tag2idx[tag]
            else:
                rv[i,j] = oov_idx
    return rv


def gen_cpy_np(ctx_word_seg_lists, ans_word_seg_lists, max_tgt_len, w2i):
    assert len(ctx_word_seg_lists) == len(ans_word_seg_lists)
    cpy_wids = np.zeros((len(ctx_word_seg_lists), max_tgt_len))
    cpy_gates = np.zeros((len(ctx_word_seg_lists), max_tgt_len))
    for bi, ctx_word_seg_list in enumerate(ctx_word_seg_lists):
        ans_word_seg_list = ans_word_seg_lists[bi]
        for ci, cw in enumerate(ctx_word_seg_list):
            if cw in w2i: continue # only allow copy for OOV words
            for ai, aw in enumerate(ans_word_seg_list):
                if cw == aw:
                    cpy_gates[bi,ai] = 1
                    cpy_wids[bi,ai] = ci
    return cpy_wids, cpy_gates
