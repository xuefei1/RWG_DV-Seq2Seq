import numpy as np
import gc
import torch
import sys
import os
import collections
import time


def merge_batch_word_seg_lists(b_seg_list_1, b_seg_list_2, remove_duplicate_words=True):
    assert len(b_seg_list_1) == len(b_seg_list_2), "Batch size must equal"
    rv = []
    for bi in range(len(b_seg_list_1)):
        if remove_duplicate_words:
            src_words = set(b_seg_list_1[bi])
            val = b_seg_list_1[bi] + [w for w in b_seg_list_2[bi] if w not in src_words]
        else:
            val = b_seg_list_1[bi] + b_seg_list_2[bi]
        rv.append(val)
    return rv


def label_tsr_to_one_hot_tsr(label_tsr, vocab_size, mask_pad=True):
    batch_size = label_tsr.shape[0]
    y_onehot = torch.zeros(batch_size, vocab_size).float()
    y_onehot.scatter_(1, label_tsr, 1)
    if mask_pad:
        y_onehot[:, 0] = 0 # mask out <PAD>
    # print(y_onehot)
    # print(y_onehot.sum())
    # print(label_tsr)
    return y_onehot


def time_checkpoint(prev_chkpt=None, chkpt_name="", verbose=True):
    curr_chkpt = time.time()
    if prev_chkpt is None:
        return curr_chkpt
    elapsed = curr_chkpt - prev_chkpt
    if verbose:
        print(chkpt_name + " elapsed {}".format(elapsed))
    return curr_chkpt


def mean(list_val, fallback_val=None):
    if len(list_val) == 0:
        if fallback_val is not None:
            return fallback_val
        else:
            raise ZeroDivisionError()
    return sum(list_val) / len(list_val)


# By @ smth
def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


def write_line_to_file(s, f_path="progress.txt", new_file=False, verbose=False):
    code = "w" if new_file else "a"
    if verbose: print(s)
    with open(f_path, code, encoding='utf-8') as f:
        f.write(s)
        f.write("\n")


def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_file_by_line(fname, skip_lines=0):
    lines = []
    with open(fname) as f:
        for i, l in enumerate(f):
            if i < skip_lines: continue
            lines.append(l)
    return lines


def split_read_process_file(in_fn, split_line_count, process_func, func_args_list, ignore_header=False):
    total_n_lines = file_len(in_fn)
    processed_lines_count = 0
    while processed_lines_count < total_n_lines:
        lines = read_file_by_line(in_fn, skip_lines=split_line_count)
        if processed_lines_count == 0 and ignore_header:
            lines = lines[1:]
        process_func(lines, *func_args_list)
        processed_lines_count += len(lines)


class UnbufferedStdOut:
    # sys.stdout=UnbufferedStdOut(sys.stdout)
    def __init__(self, stream, filename=None):
        self.stream = stream
        self.filename = filename

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if self.filename is not None:
            write_line_to_file(str(data), self.filename)


class UniqueDict(dict):
    def __init__(self, inp=None):
        self._no_dups = True
        if isinstance(inp,dict):
            super(UniqueDict,self).__init__(inp)
        else:
            super(UniqueDict,self).__init__()
            if isinstance(inp, (collections.Mapping, collections.Iterable)):
                si = self.__setitem__
                for k,v in inp:
                    si(k,v)
        self._no_dups = False

    def __setitem__(self, k, v):
        try:
            self.__getitem__(k)
            if self._no_dups:
                raise ValueError("duplicate key '{0}' found".format(k))
            else:
                super(UniqueDict, self).__setitem__(k, v)
        except KeyError:
            super(UniqueDict,self).__setitem__(k,v)
