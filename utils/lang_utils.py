import os
import gensim
import torch
import numpy as np
import collections
from constants import *
from nltk.corpus import stopwords
from utils.model_utils import device
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class W2VTrainableVocab:
    def __init__(self, w2c, w2v_dict, embedding_dim=300, light_weight=True, rand_oov_embed=False,
                 special_tokens = [PAD_TOKEN,OOV_TOKEN],
                 ):
        self.embedding_dim = embedding_dim
        self.light_weight = light_weight
        self.vocab_size = 0
        self._w2v_original = w2v_dict
        self._w2c_input = w2c
        self.oov_count = 0
        self.w2v = {}
        self.w2i = {PAD_TOKEN:0, OOV_TOKEN:1}
        self.i2w = {}
        self.i2v = {}
        self.c2i = {}
        self.i2c = {}
        self.w2v_mat = None
        self.c2v_mat = None
        self.pad_token = PAD_TOKEN
        self.oov_token = OOV_TOKEN
        self.pad_idx = 0
        self.oov_idx = 1
        self.rand_oov_embed = rand_oov_embed
        # modify this to contain all special tokens
        self.special_tokens = special_tokens
        self._curr_word_idx = len(self.w2i)
        # self.trainable_oov_token_indices = []
        self.oov_w2i = {PAD_TOKEN:self.pad_idx, OOV_TOKEN:self.oov_idx}
        self.oov_i2w = {}
        self._build_vocab()

    def _build_vocab(self):
        print("building vocab from custom w2c, vocab size: " + str(len(self._w2c_input)))
        for t in self.special_tokens:
            if len(t) == 0: continue
            if t in self.w2i: continue
            self.w2i[t] = self._curr_word_idx
            self._curr_word_idx += 1
        for w, c in self._w2c_input.items():
            if len(w) == 0: continue
            if w in self.w2i: continue
            self.w2i[w] = self._curr_word_idx
            self._curr_word_idx += 1
        self.i2w = {v: k for k, v in self.w2i.items()}
        assert len(self.w2i) == len(self.i2w)
        oov_tokens = set()
        for i, w in self.i2w.items():
            if i in self.i2v: continue
            if w in self._w2v_original:
                self.i2v[i] = self._w2v_original[w]
                self.w2v[w] = self._w2v_original[w]
            else:
                # if w == PAD_TOKEN:
                #     embed = np.zeros(self.embedding_dim)
                # elif self.rand_oov_embed:
                #     embed = np.random.randn(self.embedding_dim)
                # else:
                #     embed = np.zeros(self.embedding_dim)
                embed = np.zeros(self.embedding_dim)
                if w not in self.special_tokens:
                    self.oov_count += 1
                self.i2v[i] = embed
                self.w2v[w] = embed
                oov_tokens.add(w)

        oov_word_idx = max([i for w, i in self.oov_w2i.items()]) + 1
        for oov_w in list(oov_tokens):
            if oov_w in self.oov_w2i: continue
            self.oov_w2i[oov_w] = oov_word_idx
            oov_word_idx += 1
        self.oov_i2w = {v: k for k, v in self.oov_w2i.items()}
        assert len(self.oov_w2i) == len(self.oov_i2w)

        self._vocab_size = len(self.w2i)
        self.w2v_mat = build_i2v_mat(self._vocab_size, self.embedding_dim, self.i2v)
        self.w2v_mat = torch.from_numpy(self.w2v_mat).type(torch.FloatTensor)
        self._build_character_embedding()
        if self.light_weight:
            self.i2v = None
            self._w2v_original = None
            self._w2c_input = None
        print("{} words OOV when building vocab".format(self.oov_count))
        print("built vocab size {}".format(self._curr_word_idx))

    def _build_character_embedding(self, embedding_dim=200):
        # should be called at the end of build_vocab()
        self.c2i = {chr(i):i for i in range(128)} # TODO: for now only do ascii, and use 200d
        self.i2c = {v:k for k, v in self.c2i.items()}
        self.c2v_mat = np.random.randn(128, embedding_dim)
        self.c2v_mat = torch.from_numpy(self.c2v_mat).type(torch.FloatTensor)

    def get_oov2wi_dict(self):
        rv = {}
        for oov_idx, w in self.oov_i2w.items():
            rv[oov_idx] = self.w2i[w]
        return rv

    def get_word_idx(self, word):
        return self.w2i if word in self.w2i else self.oov_idx

    def get_word_from_idx(self, idx):
        return self.i2w if idx in self.i2w else self.oov_token

    def items(self):
        return self.w2i.items()

    def __len__(self):
        return self._vocab_size

    def __repr__(self):
        return "Vocab with additional tokens like OOV:{}, PAD:{}".format(self.oov_token, self.pad_token)


class SegData:

    def __init__(self, raw_str, seg_list,
                 pos_list=None,
                 keyword_seg_list=None,
                 keyword_pos_list=None,
                 extras={},
                 is_light_weight=False):
        self.raw_str = raw_str
        self.seg_list = seg_list
        self.pos_list = pos_list
        self.keyword_seg_list = keyword_seg_list
        self.keyword_pos_list = keyword_pos_list
        self.extras = extras
        self.is_light_weight = is_light_weight
        if pos_list is not None:
            assert len(pos_list) == len(seg_list)
        if keyword_seg_list is not None and keyword_pos_list is not None:
            assert len(keyword_seg_list) == len(keyword_pos_list)
        if self.is_light_weight:
            self.pos_list = None
            self.keyword_pos_list = None
            self.extras = None

    def __str__(self):
        return self.raw_str

    def __repr__(self):
        return self.raw_str

    def __len__(self):
        return len(self.seg_list)

    def __hash__(self):
        return hash(self.raw_str)

    def __eq__(self, other):
        if not isinstance(other, SegData):
            return False
        return self.raw_str == other.raw_str

    def __ne__(self, other):
        return not (self == other)


class WrapperW2VDict(dict):

    def __init__(self, model, model_w2i, original_w2v, oov_idx):
        super(WrapperW2VDict, self).__init__()
        self._model = model
        self._oov_idx = oov_idx
        self._model_w2i = model_w2i
        self._original_w2v = original_w2v

    def __getitem__(self, word):
        with torch.no_grad():
            if word in self._model_w2i:
                word_idx = self._model_w2i[word]
                tsr = torch.ones(1,1,1).fill_(word_idx).type(torch.LongTensor).to(device())
                embedding = self._model(tsr).squeeze().cpu().detach().numpy()
            else:
                embedding = self._original_w2v[word]
        return embedding

    def __contains__(self, item):
        return item in self._model_w2i or item in self._original_w2v


def load_gensim_word_vec(word_vec_file, cache_file=None):
    import pickle
    if cache_file is not None and os.path.isfile(cache_file):
        print("Loading word vec from cache: " + cache_file)
        with open(cache_file, "rb") as f:
            w2v = pickle.load(f)
        return w2v
    print("Loading word vec from source: " + word_vec_file)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word_vec_file, binary=False)
    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(w2v, f, protocol=4)
    return w2v


def build_w2c_from_seg_word_lists(seg_word_lists, limit=None):
    w2c = {}
    for seg_list in seg_word_lists:
        for word in seg_list:
            if len(word) == 0: continue
            if word not in w2c: w2c[word] = 0
            w2c[word] += 1
    if limit is not None and len(w2c) > limit:
        tmp = sorted([(w,c) for w,c in w2c.items()],key=lambda t:t[1],reverse=True)[:limit]
        w2c = {t[0]:t[1] for t in tmp}
    return w2c


def pad_seg_lists_in_place(seg_lists, pad_val=0):
    pad_len = max([len(seg_list) for seg_list in seg_lists])
    for seg_list in seg_lists:
        while len(seg_list) < pad_len:
            seg_list.append(pad_val)


def mean_of_w2v(word_seg_list, w2v, to_torch_tensor=False):
    word_seg_list = [w for w in word_seg_list if w in w2v]
    if len(word_seg_list) == 0:
        raise ValueError("Input is empty")
    word_vecs = [w2v[w].reshape(1,-1) for w in word_seg_list]
    word_vec = np.concatenate(word_vecs, axis=0)
    word_vec = np.mean(word_vec, axis=0, keepdims=False)
    if to_torch_tensor:
        rv = torch.from_numpy(word_vec).type(torch.FloatTensor).to(device())
    else:
        rv = word_vec
    return rv


def cosine_sim_np(np_arr_1, np_arr_2):
    cos_sim = cosine_similarity(np_arr_1.reshape(1,-1), np_arr_2.reshape(1,-1))[0,0]
    return cos_sim


def euclidean_dist_np(np_arr_1, np_arr_2):
    euc_dist = euclidean_distances(np_arr_1.reshape(1,-1), np_arr_2.reshape(1,-1))[0,0]
    return euc_dist


def gen_cpy_np(src_word_seg_lists, tgt_word_seg_lists, max_tgt_len, w2i):
    assert len(src_word_seg_lists) == len(tgt_word_seg_lists)
    cpy_wids = np.zeros((len(src_word_seg_lists), max_tgt_len))
    cpy_gates = np.zeros((len(src_word_seg_lists), max_tgt_len))
    for bi, ctx_word_seg_list in enumerate(src_word_seg_lists):
        tgt_word_seg_list = tgt_word_seg_lists[bi]
        for ci, cw in enumerate(ctx_word_seg_list):
            for ai, aw in enumerate(tgt_word_seg_list):
                if aw in w2i: continue  # only allow copy for OOV words
                if cw == aw:
                    cpy_gates[bi,ai] = 1
                    cpy_wids[bi,ai] = ci
    return cpy_wids, cpy_gates


def w2c_selective_load_w2v(w2v_file, w2c, delim=" "):
    with open(w2v_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0].split(delim)
    full_vocab_size, embedding_dim = int(header[0]), int(header[1])
    lines = lines[1:]
    w2v_words = {line.rstrip().split(delim)[0]:i for i,line in enumerate(lines) if len(line.rstrip()) > 0}
    rv_w2v = {}
    for word, _ in w2c.items():
        if word in w2v_words and word not in rv_w2v:
            line = lines[w2v_words[word]].rstrip()
            vec_vals = [float(v) for v in line.split(delim)[1:] if len(v) > 0]
            assert len(vec_vals) == embedding_dim
            word_vec = np.asarray(vec_vals)
            rv_w2v[word] = word_vec
    return rv_w2v


def select_from_d_list_by_w2c(data_list, w2c, word_key, limit):
    if len(data_list) < limit:
        return data_list
    data = sorted([(word_key(d),w2c[word_key(d)],d) for d in data_list if word_key(d) in w2c], key=lambda t:t[1], reverse=True)
    data = data[:limit]
    rv = [d[2] for d in data]
    return rv


def select_from_d_dict_by_w2c(data_dict, w2c, word_key, limit):
    if len(data_dict) < limit:
        return data_dict
    data = sorted([(word_key(k,v),w2c[word_key(k,v)], k) for k,v in data_dict.items() if word_key(k,v) in w2c], key=lambda t:t[1], reverse=True)
    data = data[:limit]
    rv = {d[2]:data_dict[d[2]] for d in data}
    return rv


def build_i2v_mat(vocab_size, embedding_dim, i2v):
    rv = np.zeros((vocab_size, embedding_dim))
    assert len(i2v) == vocab_size
    for i, v in i2v.items():
        assert isinstance(i, int)
        if i >= vocab_size: assert False, "idx {} OOV".format(i)
        rv[i, :] = i2v[i]
    return rv


def get_char_ids_tensor(seg_lists, c2i, max_char_len=16, pad_idx=0, oov_idx=1):
    """
    return batch_size x seq_len x 16
    """
    max_seq_len = max([len(l) for l in seg_lists])
    rv = np.zeros((len(seg_lists), max_seq_len, max_char_len))
    rv.fill(pad_idx)
    for i, seg_list in enumerate(seg_lists):
        for j, word in enumerate(seg_list):
            for k, c in enumerate(word):
                if k >= rv.shape[2]:
                    break
                if c in c2i:
                    rv[i,j,k] = c2i[c]
                else:
                    rv[i,j,k] = oov_idx
    return torch.from_numpy(rv).type(torch.LongTensor)


def _read_qa_seg_cache_file(path, delim=","):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    seg_rv = []
    for line in lines:
        line = line.rstrip()
        if line:
            seg_rv.append(line.split(delim))
    return seg_rv


def read_qa_corpus_file(path, read_lines_limit=None):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    raw_msg = []
    raw_rsp = []
    fetch_msg = True
    if read_lines_limit:
        print('Read lines limit: ' + str(read_lines_limit))
    pairs_count = 0
    for line in lines:
        if read_lines_limit and pairs_count >= read_lines_limit:
            break
        line = line.rstrip()
        if line:
            if fetch_msg:
                raw_msg.append(line)
            else:
                pairs_count += 1
                raw_rsp.append(line)
            fetch_msg = not fetch_msg
    return raw_msg, raw_rsp


def load_word_vectors(w2v_file, w2v_cache_file=None, read_delim=" ",):
    word2idx = {}
    import pickle
    if os.path.isfile(w2v_cache_file):
        fileObject = open(w2v_cache_file, 'rb')
        model = pickle.load(fileObject)
        fileObject.close()
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            w2v_file, binary=False)
        fileObject = open(w2v_cache_file, 'wb')
        pickle.dump(model, fileObject)
        fileObject.close()
    if hasattr(model, "index2word"):
        word2idx = {v: k for k, v in enumerate(model.index2word)}
    else:
        with open(w2v_file, encoding='utf-8') as f:
            lines = f.readlines()
        idx = 0
        for line in lines:
            tokens = line.split(read_delim)
            if len(tokens) < 10:
                continue
            word2idx[tokens[0]] = idx
            idx+=1
    idx2word = {v: k for k, v in word2idx.items()}
    return model, word2idx, idx2word


def pad_until_len(word_seg_list, target_len, pad_word="</s>"):
    for seg_list in word_seg_list:
        while len(seg_list) < target_len:
            seg_list.append(pad_word)


def one_hot_encode_word(word, word2idx, vocab_size):
    classes = np.zeros(vocab_size)
    if word in word2idx:
        idx = word2idx[word]
        classes[idx] = 1
    return classes


def _gen_vec_rep_for_tokenized_sent(word_list, max_sent_len, word_vec, embed_dim):
    rv = np.zeros((max_sent_len, embed_dim))
    i = 0
    for word in word_list:
        if i >= max_sent_len:
            break
        if word in word_vec:
            rv[i,:] = word_vec[word]
        i += 1
    return rv


def handle_OOV_for_words(words_list, dictionary, oov_token=None):
    rv = []
    for w in words_list:
        if w in dictionary:
            rv.append(w)
        elif oov_token is not None and oov_token in dictionary:
            rv.append(oov_token)
    return rv


def _read_sent_w2i_cache_file(file):
    npz_file = np.load(file)
    return npz_file['arr_0']


def remove_stop_words_en(seg_list, ):
    eng_stops = set(stopwords.words('english'))
    sent_list = [w for w in seg_list if w not in eng_stops]
    return sent_list


def _sent_word2idx_lists(seg_sent_list, word2idx, max_sent_len, oov_idx=None):
    sent_words2idx = []
    for seg_sent in seg_sent_list:
        j = 0
        tmp = []
        for word in seg_sent:
            if j >= max_sent_len:
                break
            if word in word2idx:
                tmp.append(word2idx[word])
            elif oov_idx is not None:
                tmp.append(oov_idx)
            j += 1
        sent_words2idx.append(tmp)
    return sent_words2idx


def _sent_word2idx_np(seg_sent_list, word2idx, max_sent_len, pad_idx=0, oov_idx=1):
    sent_words2idx = np.zeros((len(seg_sent_list), max_sent_len))
    if pad_idx is not None: sent_words2idx.fill(pad_idx)
    i = 0
    for seg_sent in seg_sent_list:
        j = 0
        for word in seg_sent:
            if j >= max_sent_len:
                break
            if word in word2idx:
                sent_words2idx[i,j] = word2idx[word]
            else:
                sent_words2idx[i,j] = oov_idx
            j += 1
        i += 1
    return sent_words2idx


def _sent_word2vec_np(seg_sent_list, w2v, embedding_size, max_sent_len, oov_token="<OOV>"):
    assert oov_token in w2v
    sent_words2idx = np.zeros((len(seg_sent_list), max_sent_len, embedding_size))
    i = 0
    for seg_sent in seg_sent_list:
        j = 0
        for word in seg_sent:
            if j >= max_sent_len:
                break
            if word in w2v:
                sent_words2idx[i,j,:] = w2v[word]
            else:
                sent_words2idx[i,j,:] = w2v[oov_token]
            j += 1
        i += 1
    return sent_words2idx


def sents_words2idx_to_text(w2i, w2v, eos_idx=0, oov_token="<OOV>", delim=" "):
    rv = []
    max_sent_len = get_valid_vec_rep_length(w2i.reshape(-1, 1))
    for i in range(max_sent_len):
        word_idx = int(w2i[i])
        if word_idx == eos_idx >= 0:
           break
        if 0 <= word_idx <= len(w2v.index2word):
            word = w2v.index2word[word_idx]
            rv.append(word)
        else:
            rv.append(oov_token)
    return delim.join(rv)


def sents_words2idx_to_one_hot(batch_size, w2i, vocab_size, max_sent_len_idx=1):
    max_sent_len = w2i.shape[max_sent_len_idx]
    rv = np.zeros((max_sent_len, batch_size, vocab_size))
    for i in range(batch_size):
        for j in range(max_sent_len):
            idx = int(w2i[i, j])
            if idx < 0:
                continue
            rv[j, i, idx] = 1
    return rv


def _word2idx_to_one_hot(word_idx, vocab_size):
    rv = np.zeros(vocab_size)
    rv[word_idx] = 1
    return rv


def append_to_seg_sents_list(seg_sents_list, token, to_front=True):
    for sent_word_list in seg_sents_list:
        if to_front:
            sent_word_list.insert(0, token)
        else:
            sent_word_list.append(token)


def handle_msg_rsp_OOV(msg_seg_list, rsp_seg_list, dictionary, oov_token=None):
    msg_seg_rv = []
    rsp_seg_rv = []
    for i in range(len(msg_seg_list)):
        msg_words = handle_OOV_for_words(msg_seg_list[i], dictionary, oov_token=oov_token)
        rsp_words = handle_OOV_for_words(rsp_seg_list[i], dictionary, oov_token=oov_token)
        if len(msg_words) == 0 or len(rsp_words) == 0:
            continue
        msg_seg_rv.append(msg_words)
        rsp_seg_rv.append(rsp_words)
    return msg_seg_rv, rsp_seg_rv


def get_valid_vec_rep_length(vec):
    """
    Input must be embedding_size
    """
    n_instances = vec.shape[0]
    rv = 0
    for i in range(n_instances):
        val = vec[i,:]
        if np.all(val==0) or np.all(val==-1):
            return rv
        rv += 1
    return rv


def gen_word_embedding_vec_rep(seg_sents_list, embedding_size, w2v, max_sent_len,
                               time_major=False,
                               word_embedding_vec_cache_file=None,):
    if word_embedding_vec_cache_file is not None and os.path.isfile(word_embedding_vec_cache_file):
        npz_file = np.load(word_embedding_vec_cache_file)
        return npz_file['arr_0']
    n_instances = len(seg_sents_list)
    if time_major:
        rv = np.zeros((max_sent_len, n_instances, embedding_size))
    else:
        rv = np.zeros((n_instances, max_sent_len, embedding_size))
    for i in range(n_instances):
        vec_rep = _gen_vec_rep_for_tokenized_sent(seg_sents_list[i], max_sent_len, w2v, embedding_size)
        if time_major:
            rv[:, i, :] = vec_rep
        else:
            rv[i, :, :] = vec_rep
    if word_embedding_vec_cache_file:
        np.savez(word_embedding_vec_cache_file, rv)
    return rv


def gen_word2idx_vec_rep(seg_sents_list, word2idx, max_sent_len, return_lists=False,
                         pad_idx=0, oov_idx=1):
    if return_lists:
        sent_word2idx = _sent_word2idx_lists(seg_sents_list, word2idx, max_sent_len, oov_idx)
    else:
        sent_word2idx = _sent_word2idx_np(seg_sents_list, word2idx, max_sent_len, pad_idx=pad_idx, oov_idx=oov_idx)
    return sent_word2idx


def gen_word2vec_np(seg_sents_list, w2v, embedding_size, max_sent_len, oov_token="<OOV>"):
    sent_word2vec = _sent_word2vec_np(seg_sents_list, w2v, embedding_size, max_sent_len, oov_token=oov_token)
    return sent_word2vec


def truncate_str_upto(s, upto_char, include=True):
    sp = s.split(upto_char)
    if len(sp) == 1:
        return s
    return sp[0]+upto_char if include else sp[0]

