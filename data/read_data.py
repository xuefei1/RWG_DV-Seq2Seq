from data.preprocess import *
from tqdm import tqdm
import pickle
import os
import random
from utils.misc_utils import write_line_to_file, mean
from constants import *
from utils.lang_utils import SegData, cosine_sim_np, build_w2c_from_seg_word_lists


def keywords_by_pos(word_pos_tuples, kw_pos={"nn", "vv", "nr"}, exclude_words={}):
    kw_rv = []
    pos_rv = []
    for w, tag in word_pos_tuples:
        if tag not in kw_pos or w in exclude_words:
            continue
        kw_rv.append(w)
        pos_rv.append(tag)
    return kw_rv, pos_rv


def keywords_by_pos_tw(word_pos_tw_tuples, kw_pos={"nn", "vv", "nr"}, exclude_words={}, tw_threshold=0):
    kw_rv = []
    pos_rv = []
    tw_rv = []
    for w, tag, tw in word_pos_tw_tuples:
        if tag not in kw_pos or w in exclude_words or tw <= tw_threshold:
            continue
        kw_rv.append(w)
        pos_rv.append(tag)
        tw_rv.append(tw)
    return kw_rv, pos_rv, tw_rv


def read_qry_data(in_fn, col_delim="/", w2c_qry_limit=None, word_delim=" "):
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rv = []
    qry_w2c = {}
    unique_cons = set()
    for line in lines:
        inst = line.rstrip().split(col_delim)
        qry_seg_list = inst[0].split(word_delim)
        con_seg_list = inst[1].split(word_delim)
        qry_seg_list = [w for w in qry_seg_list if len(w) > 0]
        con_seg_list = [w for w in con_seg_list if len(w) > 0]
        if len(qry_seg_list) == 0 or len(con_seg_list) == 0:
            continue
        unique_cons.add("".join(con_seg_list))
        rv.append([qry_seg_list, "".join(con_seg_list)])
        for word in qry_seg_list:
            if word not in qry_w2c: qry_w2c[word] = 0
            qry_w2c[word] += 1
    if w2c_qry_limit is not None and len(qry_w2c) > w2c_qry_limit:
        tmp = sorted([(w,c) for w,c in qry_w2c.items()],key=lambda t:t[1],reverse=True)[:w2c_qry_limit]
        qry_w2c = {t[0]:t[1] for t in tmp}
    return rv, qry_w2c, list(unique_cons)


def read_ranked_word_pair_score(in_fn, col_delim=" ", remove_same_words=True):
    rv = []
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    seg_lists = [line.rstrip().split(col_delim) for line in lines if len(line.rstrip()) > 0]
    for seg_list in seg_lists:
        seg_list = [w for w in seg_list if len(w) > 0]
        if remove_same_words and seg_list[0] == seg_list[1]:
            continue
        seg_list[-1] = float(seg_list[-1])
        rv.append(seg_list)
    max_val = max([s[2] for s in rv])
    for r in rv:
        r[2] /= (max_val/2)
        r[2] -= 1
    assert all([1 >= r[2] >= -1 for r in rv])
    return rv


def clean_chinese_query_data(in_fn, op_fn, ignore_header=True, col_delim="\t"):
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    head = 0
    for line in tqdm(lines, desc="Cleaning", ascii=True):
        line = line.rstrip().replace(" ", "")
        seg_list = line.split(col_delim)
        seg_list = [t for t in seg_list if len(t) > 0]
        if len(seg_list) != 2:
            continue
        tmp = []
        for text in seg_list:
            if no_chinese_char_found(text):
                break
            text = preprocess_chinese_raw_str(text, collapse_numbers=False)
            if len(text) < 2:
                break
            tmp.append(text)
        if len(tmp) == 2:
            rv = col_delim.join(tmp)
            write_line_to_file(rv, op_fn)
            head += 1
    print("{} lines wrote to {}".format(head, op_fn))


def build_click_dicts(seg_list):
    qry2titles, title2queries = {}, {}
    for t in tqdm(seg_list, desc="Building qry title dicts", leave=False, ascii=True):
        qry = t[0]
        title = t[1]
        if qry not in qry2titles:
            qry2titles[qry] = {}
        if title not in qry2titles[qry]:
            qry2titles[qry][title] = 0
        qry2titles[qry][title] += 1
        if title not in title2queries:
            title2queries[title] = {}
        if qry not in title2queries[title]:
            title2queries[title][qry] = 0
        title2queries[title][qry] += 1
    print("qry2titles len {}".format(len(qry2titles)))
    print("title2queries len {}".format(len(title2queries)))
    return qry2titles, title2queries


def select_representative_query(title2queries, w2c):
    def _norm_between_0_1(qry_score_pairs, one_minus=False):
        if len(qry_score_pairs) == 1:
            return [[qry_score_pairs[0][0], 1.0]]
        scores = [t[1] for t in qry_score_pairs]
        max_score = max(scores)
        min_score = min(scores)
        factor = max_score - min_score
        assert factor >= 0
        if factor == 0:
            rv = [
                [t[0], 1.0] for t in qry_score_pairs
            ]
        else:
            rv = [
                [t[0], 1.0 - (t[1] - min_score) / factor if one_minus else (t[1] - min_score) / factor]
                for t in qry_score_pairs
            ]
        return rv
    pairs = []
    unique_reps = {}
    unique_qrys = {}
    uniques = set()
    for title, qs in title2queries.items():
        if len(qs) <= 1: continue
        q_count_scores = []
        q_len_scores = []
        q_aggr_scores = {}
        q_list = [qry for qry, _ in qs.items()]
        lens = [len(q) for q in q_list]
        for qry in q_list:
            if len(qry) == 0 or qry in unique_qrys: continue
            w2c_scores = [w2c[w] if w in w2c else 1 for w in qry.keyword_seg_list]
            if len(w2c_scores) == 0: continue
            q_count_scores.append( [qry, mean(w2c_scores)] )
            q_len_scores.append( [qry, len(qry)] )
            unique_qrys[qry] = 1
        if len(q_count_scores) <= 1 or len(q_len_scores) <= 1: continue
        q_count_scores = _norm_between_0_1(q_count_scores)
        q_len_scores = _norm_between_0_1(q_len_scores, one_minus=True)
        for qry, score in q_count_scores:
            if qry in q_aggr_scores: continue
            q_aggr_scores[qry] = 0.25 * score
        for qry, score in q_len_scores:
            if qry not in q_aggr_scores: continue
            q_aggr_scores[qry] += 0.75 * score
        if len(q_aggr_scores) == 0: continue
        sorted_scores = sorted([(k,v) for k,v in q_aggr_scores.items()], key=lambda t:t[1], reverse=True)
        rep = sorted_scores[0][0]
        if rep not in unique_reps:
            unique_reps[rep] = 0
        unique_reps[rep] += 1
        for qry in q_list:
            if rep == qry: continue
            key = qry.raw_str + "|" + rep.raw_str
            if key in uniques: continue
            uniques.add(key)
            pairs.append([qry, rep])
    print("{} qry tgt pairs selected".format(len(pairs)))
    print("{} unique representatives".format(len(unique_reps)))
    return pairs


def get_queries_seg_data(in_fn, ignore_header=False, col_delim="\t", cache_file=None):
    from stanfordcorenlp import StanfordCoreNLP
    d = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".." + os.path.sep + "libs"+ os.path.sep
    nlp = StanfordCoreNLP(d + "stanford-corenlp-full-2018-10-05", lang='zh')
    kw_pos = {"nn", "nnp", "nnps", "vv", "jj", "nr", "ner", "vb"}
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    rv = []
    for line in tqdm(lines, desc="Segmenting", leave=False, ascii=True):
        seg_list = line.split(col_delim)
        if len(seg_list) == 2:
            s1, s2 = seg_list
            s1, s2 = s1.rstrip(), s2.rstrip()
            seg1, seg2 = nlp.word_tokenize(s1.lower()), nlp.word_tokenize(s2.lower())
            pos1, pos2 = nlp.pos_tag(s1), nlp.pos_tag(s2)
            pos1 = [t[1].lower() for t in pos1]
            pos2 = [t[1].lower() for t in pos2]
            kws1, kwpos1 = keywords_by_pos(list(zip(seg1, pos1)), kw_pos=kw_pos)
            kws2, kwpos2 = keywords_by_pos(list(zip(seg2, pos2)), kw_pos=kw_pos)
            assert len(pos1) == len(seg1) and len(pos2) == len(seg2)
            seg_data_list = [
                SegData(s1, seg1, pos_list=pos1, keyword_seg_list=kws1, keyword_pos_list=kwpos1),
                SegData(s2, seg2, pos_list=pos2, keyword_seg_list=kws2, keyword_pos_list=kwpos2)
            ]
            rv.append(seg_data_list)
    if cache_file is not None:
        pickle.dump(rv, open(cache_file, "wb"), protocol=4)
    return rv


def read_pre_seg_data(in_fn, ignore_header=False, word_tag_delim="0101010", token_delim="/", col_delim="\t",
                      cache_file=None, expected_n_cols=2, read_lines_limit=None):
    if cache_file is not None and os.path.isfile(cache_file):
        rv = pickle.load(open(cache_file, "rb"))
        print("Reading cached data from {}".format(cache_file))
        return rv
    kw_pos = {"n", "nr", "v", "vn", "ns", "w", "nt", "vi", "f", "company", "nit", "nz"}
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    lines = lines[:read_lines_limit] if read_lines_limit is not None else lines
    rv = []
    for line in tqdm(lines, desc="Reading", leave=False, ascii=True):
        line = line.rstrip()
        seg_list = line.split(col_delim)
        if len(seg_list) == expected_n_cols:
            qry, title = seg_list[0], seg_list[1]
            qry_pairs = qry.split(token_delim)
            title_pairs = title.split(token_delim)
            qry_words, qry_tags = [], []
            title_words, title_tags = [], []
            valid = True
            for pair in qry_pairs:
                pair_seg = pair.split(word_tag_delim)
                if len(pair_seg) != 2: valid = False
                word = pair_seg[0]
                tag = pair_seg[1]
                if word == "" and tag == "w": word = "#"
                qry_words.append(word)
                qry_tags.append(tag)
            for pair in title_pairs:
                pair_seg = pair.split(word_tag_delim)
                if len(pair_seg) != 2: valid = False
                word = pair_seg[0]
                tag = pair_seg[1]
                if word == "" and tag == "w": word = "#"
                title_words.append(word)
                title_tags.append(tag)
            if len(qry_words) == 0 or len(title_words) == 0 or not valid: continue
            qry_tags = [t.lower() for t in qry_tags]
            title_tags = [t.lower() for t in title_tags]
            kws_qry, kwpos_qry = keywords_by_pos(list(zip(qry_words, qry_tags)), kw_pos=kw_pos)
            kws_title, kwpos_title = keywords_by_pos(list(zip(title_words, title_tags)), kw_pos=kw_pos)
            seg_data_list = [
                SegData("".join(qry_words), qry_words, pos_list=qry_tags, keyword_seg_list=kws_qry, keyword_pos_list=kwpos_qry),
                SegData("".join(title_words), title_words, pos_list=title_tags, keyword_seg_list=kws_title, keyword_pos_list=kwpos_title),
            ]
            rv.append(seg_data_list)
        else:
            print("Warning: segmented column numbers do not match expected value {}".format(expected_n_cols))
    if cache_file is not None:
        random.shuffle(rv)
        pickle.dump(rv, open(cache_file, "wb"), protocol=4)
        print("Data cached to {}".format(cache_file))
    return rv


def read_pre_seg_tw_data(in_fn, ignore_header=False, word_tag_delim="/",
                         tag_tw_delim=":", token_delim=" ", col_delim="|",
                         cache_file=None, expected_n_cols=9, read_lines_limit=None,
                         light_weight=False):
    if cache_file is not None and os.path.isfile(cache_file):
        rv = pickle.load(open(cache_file, "rb"))
        print("Read {} lines of cached data from {}".format(len(rv), cache_file))
        return rv
    kw_pos = load_kw_pos()
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    lines = lines[:read_lines_limit] if read_lines_limit is not None else lines
    rv = []
    for line in tqdm(lines, desc="Building objects", leave=False, ascii=True):
        line = line.rstrip()
        seg_list = line.split(col_delim)
        if len(seg_list) == expected_n_cols:
            qry, title = seg_list[5], seg_list[7]
            qry_tokens = qry.split(token_delim)
            title_tokens = title.split(token_delim)
            qry_words, qry_tags, qry_tws = [], [], []
            title_words, title_tags, title_tws = [], [], []
            for tokens in qry_tokens:
                token_seg = re.split(word_tag_delim+"|"+tag_tw_delim, tokens)
                if len(token_seg) != 3:
                    continue
                word = token_seg[0]
                tag = token_seg[1]
                tw = float(token_seg[2])
                qry_words.append(word)
                qry_tags.append(tag)
                qry_tws.append(tw)
            for tokens in title_tokens:
                token_seg = re.split(word_tag_delim+"|"+tag_tw_delim, tokens)
                if len(token_seg) != 3:
                    continue
                word = token_seg[0]
                tag = token_seg[1]
                tw = float(token_seg[2])
                title_words.append(word)
                title_tags.append(tag)
                title_tws.append(tw)
            if len(qry_words) == 0 or len(title_words) == 0: continue
            qry_tags = [t.lower() for t in qry_tags]
            title_tags = [t.lower() for t in title_tags]
            kws_qry, kw_pos_qry, kw_tws_qry = keywords_by_pos_tw(list(zip(qry_words, qry_tags, qry_tws)), kw_pos=kw_pos)
            kws_title, kw_pos_title, kw_tws_title = keywords_by_pos_tw(list(zip(title_words, title_tags, title_tws)), kw_pos=kw_pos)
            seg_data_list = [
                SegData("".join(qry_words), qry_words, pos_list=qry_tags, keyword_seg_list=kws_qry, keyword_pos_list=kw_pos_qry,
                        extras={SD_DK_TW_LIST:qry_tws, SD_DK_KW_TW_LIST:kw_tws_qry}, is_light_weight=light_weight),
                SegData("".join(title_words), title_words, pos_list=title_tags, keyword_seg_list=kws_title, keyword_pos_list=kw_pos_title,
                        extras={SD_DK_TW_LIST: title_tws, SD_DK_KW_TW_LIST: kw_tws_title}, is_light_weight=light_weight),
            ]
            rv.append(seg_data_list)
        else:
            print("Warning: segmented column numbers do not match expected value {}".format(expected_n_cols))
    if cache_file is not None:
        random.shuffle(rv)
        pickle.dump(rv, open(cache_file, "wb"), protocol=4)
        print("{} lines of data cached to {}".format(len(rv), cache_file))
    return rv


def gen_qry_log_files(files_list, ignore_header=False, read_lines_limit=None):
    for i in range(len(files_list)):
        qry_log_file = files_list[i]
        with open(qry_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = lines[1:] if ignore_header else lines
        lines = lines[:read_lines_limit] if read_lines_limit is not None else lines
        yield gen_pre_seg_tw_data(lines, light_weight=True)


def write_simplified_click_graph(files_list, output_file,
                                 word_tag_delim="/",tag_tw_delim=":",
                                 token_delim=" ", col_delim="|"):
    kw_pos = load_kw_pos()
    write_head = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for file in files_list:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in tqdm(lines, desc="working on file {}".format(file)):
                line = line.rstrip()
                seg_list = line.split(col_delim)
                qry, title = seg_list[5], seg_list[7]
                qry_tokens = qry.split(token_delim)
                title_tokens = title.split(token_delim)
                qry_words, qry_tags, qry_tws = [], [], []
                title_words, title_tags, title_tws = [], [], []
                for tokens in qry_tokens:
                    token_seg = re.split(word_tag_delim + "|" + tag_tw_delim, tokens)
                    if len(token_seg) != 3:
                        continue
                    word = token_seg[0]
                    tag = token_seg[1]
                    tw = float(token_seg[2])
                    qry_words.append(word)
                    qry_tags.append(tag)
                    qry_tws.append(tw)
                for tokens in title_tokens:
                    token_seg = re.split(word_tag_delim + "|" + tag_tw_delim, tokens)
                    if len(token_seg) != 3:
                        continue
                    word = token_seg[0]
                    tag = token_seg[1]
                    tw = float(token_seg[2])
                    title_words.append(word)
                    title_tags.append(tag)
                    title_tws.append(tw)
                if len(qry_words) == 0 or len(title_words) == 0: continue
                qry_tags = [t.lower() for t in qry_tags]
                title_tags = [t.lower() for t in title_tags]
                kws_qry, kw_pos_qry, kw_tws_qry = keywords_by_pos_tw(list(zip(qry_words, qry_tags, qry_tws)),
                                                                     kw_pos=kw_pos)
                kws_title, kw_pos_title, kw_tws_title = keywords_by_pos_tw(
                    list(zip(title_words, title_tags, title_tws)), kw_pos=kw_pos)
                if len(kws_qry) == 0 or len(kws_title) == 0: continue
                new_line = [" ".join(qry_words), " ".join(kws_qry), " ".join(title_words), " ".join(kws_title)]
                out_f.write("|".join(new_line) + "\n")
                write_head += 1
    print("simplified click graph: {} lines wrote to {}".format(write_head, output_file))


def click_graph_generator(lines, col_delim="|", word_delim=" "):
    for line in lines:
        line = line.rstrip()
        qry_ws, qry_kws, title_ws, title_kws = line.split(col_delim)
        qry_words = qry_ws.split(word_delim)
        qry_keywords = qry_kws.split(word_delim)
        title_words = title_ws.split(word_delim)
        title_keywords = title_kws.split(word_delim)
        seg_data_list = [
            SegData("".join(qry_words), qry_words, keyword_seg_list=qry_keywords, is_light_weight=True),
            SegData("".join(title_words), title_words, keyword_seg_list=title_keywords, is_light_weight=True)
        ]
        yield seg_data_list


def read_click_graph_data(lines, col_delim="|", word_delim=" "):
    rv = []
    for line in lines:
        line = line.rstrip()
        qry_ws, qry_kws, title_ws, title_kws = line.split(col_delim)
        qry_words = qry_ws.split(word_delim)
        qry_keywords = qry_kws.split(word_delim)
        title_words = title_ws.split(word_delim)
        title_keywords = title_kws.split(word_delim)
        seg_data_list = [
            SegData("".join(qry_words), qry_words, keyword_seg_list=qry_keywords, is_light_weight=True),
            SegData("".join(title_words), title_words, keyword_seg_list=title_keywords, is_light_weight=True)
        ]
        rv.append(seg_data_list)
    return rv


def read_compressed_click_graph_data(lines, col_delim="|", word_delim=" "):
    rv = []
    for line in tqdm(lines, desc="Building compressed click graph data", ascii=True):
        line = line.rstrip()
        qry_ws, title_ws, qry_kws, title_kws, counts = line.split(col_delim)
        qry_words = qry_ws.split(word_delim)
        qry_keywords = qry_kws.split(word_delim)
        title_words = title_ws.split(word_delim)
        title_keywords = title_kws.split(word_delim)
        seg_data_list = [
            SegData("".join(qry_words), qry_words, keyword_seg_list=qry_keywords, is_light_weight=True),
            SegData("".join(title_words), title_words, keyword_seg_list=title_keywords, is_light_weight=True),
            int(counts)
        ]
        rv.append(seg_data_list)
    return rv


def build_click_dicts_from_compressed_data(seg_list):
    qry2titles, title2queries = {}, {}
    for t in tqdm(seg_list, desc="Building qry title dicts", leave=False, ascii=True):
        qry = t[0]
        title = t[1]
        count = t[2]
        if qry not in qry2titles:
            qry2titles[qry] = {}
        if title not in qry2titles[qry]:
            qry2titles[qry][title] = count
        if title not in title2queries:
            title2queries[title] = {}
        if qry not in title2queries[title]:
            title2queries[title][qry] = count
    print("qry2titles len {}".format(len(qry2titles)))
    print("title2queries len {}".format(len(title2queries)))
    return qry2titles, title2queries


def build_wi_qrw_data_from_click_graph(fn="click_graph.txt"):
    with open(fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    seg_list = read_compressed_click_graph_data(lines)
    q2t, t2q = build_click_dicts_from_compressed_data(seg_list)
    report_click_graph_nth_link_weight_ratio(q2t)
    q2t, t2q = filter_contradicting_qrys(q2t, t2q, topk=2)
    # seg_list = seg_list[:1000]
    seg_list = [[t[0], t[1]] for t in seg_list]
    _w2c = build_w2c_from_seg_word_lists([qry.seg_list + title.seg_list for qry, title in seg_list])
    k_hop_data = find_upto_k_hop_qrys(seg_list, q2t, t2q, k=2)
    build_k_hop_qrw_wi_data_from_click_graph(k_hop_data, q2t, _w2c, qrw_k=2, wi_k=2)


def load_chinese_stop_words_set(fn="stop_words_zh.txt"):
    rv = set()
    with open(fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if len(line) > 0:
            rv.add(line)
    return rv


def percent_word_overlap_score(qry, titles, exclude_words_set):
    qry_words = set([w for w in qry.seg_list if w not in exclude_words_set])
    title_words = set()
    for title in titles:
        for w in title.seg_list:
            if w not in exclude_words_set:
                title_words.add(w)
    total_words = qry_words.union(title_words)
    overlap_words = set()
    for w in qry_words:
        if w in title_words:
            overlap_words.add(w)
    if len(qry_words) == 0: return 0.0
    return len(overlap_words) / len(total_words)


def build_k_hop_qrw_wi_data_from_click_graph(k_hop_seg_list, q2t, w2c, qrw_k, wi_k,
                                             wi_limit=500, valid_ratio=0.05, test_ratio=0.05):

    def _norm_between_0_1(qry_score_pairs, one_minus=False):
        if len(qry_score_pairs) == 1:
            return [[qry_score_pairs[0][0], 1.0]]
        scores = [t[1] for t in qry_score_pairs]
        max_score = max(scores)
        min_score = min(scores)
        factor = max_score - min_score
        assert factor >= 0
        if factor == 0:
            r = [
                [t[0], 1.0] for t in qry_score_pairs
            ]
        else:
            r = [
                [t[0], 1.0 - (t[1] - min_score) / factor if one_minus else (t[1] - min_score) / factor]
                for t in qry_score_pairs
            ]
        return r

    write_head = 0
    random.shuffle(k_hop_seg_list)
    valid_amount = int(len(k_hop_seg_list) * valid_ratio)
    test_amount = int(len(k_hop_seg_list) * test_ratio)
    stop_words_set = load_chinese_stop_words_set()
    stop_words = list(stop_words_set)
    print("approximately {} lines of train data".format(len(k_hop_seg_list)-valid_amount-test_amount))
    print("approximately {} lines of valid data".format(valid_amount))
    print("approximately {} lines of test data".format(test_amount))
    f_train = open("qrw_wi_train.txt", "w", encoding="utf-8")
    f_valid = open("qrw_wi_valid.txt", "w", encoding="utf-8")
    f_test = open("qrw_wi_test.txt", "w", encoding="utf-8")
    unique_lines = set()
    for data in tqdm(k_hop_seg_list, desc="Writing qrw and wi data", ascii=True):
        curr_qry = data[0]
        curr_title = data[1]
        k_hop_q_siblings = data[2]
        k_hop_t_siblings = data[3]
        cand_qrys = [curr_qry]
        for i in range(qrw_k):
            if i >= len(k_hop_q_siblings): break
            cand_qrys += k_hop_q_siblings[i]
        cand_qrys = list(set(cand_qrys))
        if len(cand_qrys) > 100: continue # Too many equal queries usually indicate problems
        if len(cand_qrys) <= 1: continue
        cand_titles = [curr_title]
        for i in range(qrw_k):
            if i >= len(k_hop_t_siblings): break
            cand_titles += k_hop_t_siblings[i]
        cand_titles = list(set(cand_titles))
        qry_click_counts = {}
        for qry in cand_qrys:
            if qry not in q2t:
                qry_click_counts[qry] = 1
            else:
                q_count = sum([c for _, c in q2t[qry].items()])
                qry_click_counts[qry] = q_count
        q_click_count_scores = [[q, c] for q, c in qry_click_counts.items()]
        q_w2c_scores = []
        q_len_scores = []
        for qry in cand_qrys:
            if len(qry) == 0: continue
            w2c_scores = [w2c[w] if w in w2c else 1 for w in qry.seg_list if w not in stop_words_set and len(w) > 1]
            if len(w2c_scores) == 0: continue
            q_w2c_scores.append( [qry, mean(w2c_scores)] )
            q_len_scores.append( [qry, len(qry)] )
        if len(q_w2c_scores) <= 1 or len(q_len_scores) <= 1: continue
        q_w2c_scores = _norm_between_0_1(q_w2c_scores)
        q_click_count_scores = _norm_between_0_1(q_click_count_scores)
        q_len_scores = _norm_between_0_1(q_len_scores, one_minus=True)
        qt_overlaps_scores = [[q, percent_word_overlap_score(q, cand_titles, stop_words_set)] for q in cand_qrys]
        q_aggr_scores = {}
        for qry, score in q_w2c_scores:
            if qry in q_aggr_scores: continue
            q_aggr_scores[qry] = 0.0 * score
        for qry, score in q_len_scores:
            if qry not in q_aggr_scores: continue
            q_aggr_scores[qry] += 0.30 * score
        for qry, score in q_click_count_scores:
            if qry not in q_aggr_scores: continue
            q_aggr_scores[qry] += 0.40 * score
        for qry, score in qt_overlaps_scores:
            if qry not in q_aggr_scores: continue
            q_aggr_scores[qry] += 0.30 * score
        if len(q_aggr_scores) == 0: continue
        sorted_scores = sorted([(k,v) for k,v in q_aggr_scores.items()], key=lambda t:t[1], reverse=True)
        rep = sorted_scores[0][0]
        if rep == curr_qry: continue

        # related words
        cands = [curr_qry]
        for i in range(wi_k):
            if i >= len(k_hop_q_siblings): break
            cands += k_hop_q_siblings[i]
        cands = list(set(cands))
        cand_words = set()
        for cand in cands:
            for word in cand.keyword_seg_list:
                if word in stop_words_set: continue
                if len(word) <= 1: continue
                if any([word in w or w in word for w in stop_words]): continue
                cand_words.add(word)
        cand_words = list(cand_words)
        if len(cand_words) > wi_limit:
            cand_word_w2c_scores = []
            for word in cand_words:
                cand_word_w2c_scores.append([word, w2c[word] if word in w2c else 1])
            cand_word_w2c_scores = sorted(cand_word_w2c_scores, key=lambda t:t[1], reverse=True)[:wi_limit]
            cand_words = [w for w, _ in cand_word_w2c_scores]

        if len(curr_qry) == 0 or len(curr_title) == 0 or len(rep) == 0 or len(cand_words) <= 1:
            continue
        cols = [" ".join(curr_qry.seg_list), " ".join(curr_title.seg_list), " ".join(rep.seg_list), " ".join(cand_words)]
        line = "|".join(cols) + "\n"
        if line in unique_lines:
            continue
        unique_lines.add(line)
        if write_head < valid_amount:
            f_valid.write(line)
        elif valid_amount <= write_head < valid_amount + test_amount:
            f_test.write(line)
        else:
            f_train.write(line)
        write_head += 1

    f_train.close()
    f_valid.close()
    f_test.close()


def extract_wi_pairs_from_qrw_wi_data(data_seg_list):
    unique_pairs = set()
    input_n_words = []
    output_n_words = []
    overlap_n_words = []
    n_input_words = set()
    n_output_words = set()
    rv = []
    for row in data_seg_list:
        rv.append([row[0], row[3]])
        key = "|".join([" ".join(row[0]), " ".join(row[3])])
        if key not in unique_pairs:
            unique_pairs.add(key)
            input_n_words.append(len(row[0]))
            output_n_words.append(len(row[3]))
            overlap_n_words.append(len(set([w for w in row[3] + row[0] if w in row[3] and w in row[0]])))
            for w in row[0]: n_input_words.add(w)
            for w in row[3]: n_output_words.add(w)
    print("unique wi pairs: {}".format(len(unique_pairs)))
    print("avg input n words: {}".format(mean(input_n_words)))
    print("avg output n words: {}".format(mean(output_n_words)))
    print("avg overlapping n words: {}".format(mean(overlap_n_words)))
    print("n input words: {}".format(len(n_input_words)))
    print("n output words: {}".format(len(n_output_words)))
    print("")
    return rv


def extract_qrw_pairs_from_qrw_wi_data(data_seg_list):
    rv = []
    for row in data_seg_list:
        rv.append([row[0], row[2]])
    return rv


def extract_qrw_qg_triplets_from_qrw_wi_data(data_seg_list):
    rv = []
    for row in data_seg_list:
        rv.append([row[0], row[1], row[2]])
    return rv


def extract_qrw_pre_gen_wi_triplets(data_seg_list):
    unique_pairs = set()
    input_n_words = []
    output_n_words = []
    overlap_n_words = []
    n_input_words = set()
    n_output_words = set()
    rv = []
    for row in data_seg_list:
        rv.append([row[0], row[2], row[3]])
        key = "|".join([" ".join(row[0]), " ".join(row[2])])
        if key not in unique_pairs:
            unique_pairs.add(key)
            input_n_words.append(len(row[0]))
            output_n_words.append(len(row[2]))
            overlap_n_words.append(len(set([w for w in row[2] + row[0] if w in row[2] and w in row[0]])))
            for w in row[0]: n_input_words.add(w)
            for w in row[2]: n_output_words.add(w)
    print("unique pairs: {}".format(len(unique_pairs)))
    print("avg input n words: {}".format(mean(input_n_words)))
    print("avg output n words: {}".format(mean(output_n_words)))
    print("avg overlapping n words: {}".format(mean(overlap_n_words)))
    print("n input words: {}".format(len(n_input_words)))
    print("n output words: {}".format(len(n_output_words)))
    print("")
    return rv


def extract_qg_pre_gen_wi_triplets(data_seg_list):
    unique_pairs = set()
    input_n_words = []
    output_n_words = []
    overlap_n_words = []
    n_input_words = set()
    n_output_words = set()
    rv = []
    for row in data_seg_list:
        rv.append([row[1], row[2], row[4]])
        key = "|".join([" ".join(row[1]), " ".join(row[2])])
        if key not in unique_pairs:
            unique_pairs.add(key)
            input_n_words.append(len(row[1]))
            output_n_words.append(len(row[2]))
            overlap_n_words.append(len(set([w for w in row[2] + row[1] if w in row[2] and w in row[1]])))
            for w in row[1]: n_input_words.add(w)
            for w in row[2]: n_output_words.add(w)
    print("unique pairs: {}".format(len(unique_pairs)))
    print("avg input n words: {}".format(mean(input_n_words)))
    print("avg output n words: {}".format(mean(output_n_words)))
    print("avg overlapping n words: {}".format(mean(overlap_n_words)))
    print("n input words: {}".format(len(n_input_words)))
    print("n output words: {}".format(len(n_output_words)))
    print("")
    return rv


def extract_qg_pairs_from_qrw_wi_data(data_seg_list):
    unique_pairs = set()
    rv = []
    for row in data_seg_list:
        unique_pairs.add("|".join([" ".join(row[1]), " ".join(row[2])]))
        rv.append([row[1], row[2]])
    print("unique qg pairs: {}".format(len(unique_pairs)))
    return rv


def report_click_count_graph(compressed_seg_list, bins=500):
    import matplotlib.pyplot as plt
    counts = [0 for _ in range(bins)]
    for data in compressed_seg_list:
        if data[2] < len(counts):
            counts[data[2]] += 1
        else:
            counts[-1] += 1
    # counts = [data[2] for data in compressed_seg_list]
    plt.hist(counts)
    plt.xlabel('Counts')
    plt.ylabel('Number of Instances')
    plt.title('Click count distribution')
    plt.show()


def report_click_graph_nth_link_weight_ratio(q2t, n=10, terminate=False):
    # import matplotlib.pyplot as plt
    import numpy as np
    # n-th link weight
    link_weights = [ [] for _ in range(n-1)]
    n_out_links = [ 0 for _ in range(10)]
    for qry, ts in tqdm(q2t.items(), desc="Counting n-th link weights", ascii=True):
        sorted_scores = sorted([(t, c) for t, c in ts.items()], key=lambda t:t[1], reverse=True)
        n_outs = len(sorted_scores)
        if n_outs >= len(n_out_links):
            continue
            # n_outs = len(n_out_links)
        n_out_links[n_outs] += 1
        for i, tup_curr in enumerate(sorted_scores[:n]):
            if i == 0: continue
            tup_prev = sorted_scores[0]
            _, count_prev = tup_prev
            _, count_curr = tup_curr
            ratio = count_curr / count_prev
            assert 0 <= ratio <= 1.0
            link_weights[i].append(ratio)

    link_weights = [np.asarray(vals) if len(vals) > 0 else None for vals in link_weights ]
    lw_mean = [np.mean(vals) if vals is not None else 0 for vals in link_weights]
    lw_var = [np.std(vals) if vals is not None else 0 for vals in link_weights]
    print("out links distribution {}".format(str(n_out_links)))
    print("out links distribution ratio {}".format(str([n / len(q2t) for n in n_out_links])))
    print("ratio_mean {}".format(str(lw_mean)))
    print("ratio_std {}".format(str(lw_var)))
    if terminate: exit(0)


def find_upto_k_hop_qrys(seg_list, q2t, t2q, k=3):

    def _get_qrys_by_titles(t_list, exclude_qs):
        r = set()
        for t in t_list:
            qs = t2q[t]
            for q, _ in qs.items():
                if q in exclude_qs: continue
                r.add(q)
        return list(r)

    def _recur_find_k_hop_qrys(q_list, curr_k, qs_list, ts_list, visited_qs, visited_ts):
        k_hop_titles = set()
        for q in q_list:
            visited_qs.add(q)
            titles = q2t[q]
            for t, c in titles.items():
                if t in visited_ts: continue
                k_hop_titles.add(t)
                visited_ts.add(t)
        k_hop_titles = list(k_hop_titles)
        sib_qrys = _get_qrys_by_titles(k_hop_titles, visited_qs)
        for q in sib_qrys:
            qs_list[curr_k-1].add(q)
        for t in k_hop_titles:
            ts_list[curr_k-1].add(t)
        if curr_k < k:
            _recur_find_k_hop_qrys(sib_qrys, curr_k + 1, qs_list, ts_list, visited_qs, visited_ts)

    rv = []
    for data in tqdm(seg_list, desc="Building k hop data", ascii=True):
        qry = data[0]
        title = data[1]
        k_hop_q_siblings = [set() for _ in range(k)]
        k_hop_t_siblings = [set() for _ in range(k)]
        if title not in t2q or len(t2q[title]) <= 1: continue # only one query points to this title
        _recur_find_k_hop_qrys([qry], 1, k_hop_q_siblings, k_hop_t_siblings, visited_qs=set(), visited_ts=set())
        rv.append([qry, title, [list(s) for s in k_hop_q_siblings], [list(s) for s in k_hop_t_siblings]])

    n_k_hop_non_empty = [0 for _ in range(k)]
    for data in rv:
        k_hop_sibs = data[2]
        for i in range(k):
            if len(k_hop_sibs[i]) > 0:
                n_k_hop_non_empty[i] += 1

    for i in range(k):
        print("{}% of {} hop siblings non-empty".format(n_k_hop_non_empty[i] / len(rv) * 100, i+1))

    return rv


def gen_pre_seg_tw_data(lines, word_tag_delim="/",tag_tw_delim=":",
                        token_delim=" ", col_delim="|",
                        expected_n_cols=9,
                        light_weight=False):
    kw_pos = load_kw_pos()
    for line in lines:
        line = line.rstrip()
        seg_list = line.split(col_delim)
        if len(seg_list) == expected_n_cols:
            qry, title = seg_list[5], seg_list[7]
            qry_tokens = qry.split(token_delim)
            title_tokens = title.split(token_delim)
            qry_words, qry_tags, qry_tws = [], [], []
            title_words, title_tags, title_tws = [], [], []
            for tokens in qry_tokens:
                token_seg = re.split(word_tag_delim+"|"+tag_tw_delim, tokens)
                if len(token_seg) != 3:
                    continue
                word = token_seg[0]
                tag = token_seg[1]
                tw = float(token_seg[2])
                qry_words.append(word)
                qry_tags.append(tag)
                qry_tws.append(tw)
            for tokens in title_tokens:
                token_seg = re.split(word_tag_delim+"|"+tag_tw_delim, tokens)
                if len(token_seg) != 3:
                    continue
                word = token_seg[0]
                tag = token_seg[1]
                tw = float(token_seg[2])
                title_words.append(word)
                title_tags.append(tag)
                title_tws.append(tw)
            if len(qry_words) == 0 or len(title_words) == 0: continue
            qry_tags = [t.lower() for t in qry_tags]
            title_tags = [t.lower() for t in title_tags]
            kws_qry, kw_pos_qry, kw_tws_qry = keywords_by_pos_tw(list(zip(qry_words, qry_tags, qry_tws)), kw_pos=kw_pos)
            kws_title, kw_pos_title, kw_tws_title = keywords_by_pos_tw(list(zip(title_words, title_tags, title_tws)), kw_pos=kw_pos)
            seg_data_list = [
                SegData("".join(qry_words), qry_words, pos_list=qry_tags, keyword_seg_list=kws_qry, keyword_pos_list=kw_pos_qry,
                        extras={SD_DK_TW_LIST:qry_tws, SD_DK_KW_TW_LIST:kw_tws_qry}, is_light_weight=light_weight),
                SegData("".join(title_words), title_words, pos_list=title_tags, keyword_seg_list=kws_title, keyword_pos_list=kw_pos_title,
                        extras={SD_DK_TW_LIST: title_tws, SD_DK_KW_TW_LIST: kw_tws_title}, is_light_weight=light_weight),
            ]
            yield seg_data_list
        else:
            print("Warning: segmented column numbers do not match expected value {}".format(expected_n_cols))


def read_col_word_delim_data(in_fn, ignore_header=False, col_delim="|", word_delim=" "):
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    rv = []
    for line in lines:
        line = line.rstrip()
        if len(line) == 0: continue
        segs = [[w for w in d.split(word_delim) if len(w) > 0] for d in line.split(col_delim)]
        assert len(segs) >= 2
        rv.append(segs)
    return rv


def bucket_sample_dev_test(all_data, label2idx, valid_ratio=0.05, test_ratio=0.05):
    data_train, data_valid, data_test = [], [], []
    keys_to_next_idx_in_v = {}
    taken_indices = {}
    for k, v in label2idx.items():
        random.shuffle(v)
    valid_amount = len(all_data) * valid_ratio
    test_amount = len(all_data) * test_ratio
    watchdog_count = 0
    watchdog = valid_amount * 10
    while len(data_valid) < valid_amount and watchdog_count < watchdog:
        for k, v in label2idx.items():
            if len(v) < 3: continue
            if k not in keys_to_next_idx_in_v:
                keys_to_next_idx_in_v[k] = 0
            next_idx_in_v = keys_to_next_idx_in_v[k]
            if next_idx_in_v < len(v):
                next_idx = v[next_idx_in_v]
                if next_idx in taken_indices: assert False
                taken_indices[next_idx] = ""
                data_valid.append(all_data[next_idx])
                keys_to_next_idx_in_v[k] += 1
                if len(data_valid) >= valid_amount:
                    break
            watchdog_count += 1
    if watchdog_count == watchdog: assert False, "watchdog triggered when sampling valid set"
    watchdog_count = 0
    watchdog = test_amount * 10
    while len(data_test) < test_amount and watchdog_count < watchdog:
        for k, v in label2idx.items():
            if len(v) < 3: continue
            if k not in keys_to_next_idx_in_v:
                keys_to_next_idx_in_v[k] = 0
            next_idx_in_v = keys_to_next_idx_in_v[k]
            if next_idx_in_v < len(v):
                next_idx = v[next_idx_in_v]
                if next_idx in taken_indices: assert False
                taken_indices[next_idx] = ""
                data_test.append(all_data[next_idx])
                keys_to_next_idx_in_v[k] += 1
                if len(data_test) >= test_amount:
                    break
        watchdog_count += 1
    if watchdog_count == watchdog: assert False, "watchdog triggered when sampling test set"
    for di, d in enumerate(all_data):
        if di not in taken_indices:
            data_train.append(d)
    random.shuffle(data_train)
    random.shuffle(data_valid)
    random.shuffle(data_test)
    return data_train, data_valid, data_test


def censor_prep(f_in, f_out, word_tag_delim="/", tag_tw_delim=":", token_delim=" ", col_delim="|"):
    with open(f_in, "r", encoding="utf-8") as f:
        lines = f.readlines()
    line_id = 0
    with open(f_out, "w", encoding="utf-8") as f:
        for line in tqdm(lines, desc="Writing", leave=False, ascii=True):
            line = line.rstrip()
            seg_list = line.split(col_delim)
            qry, title = seg_list[5], seg_list[7]
            qry_tokens = qry.split(token_delim)
            title_tokens = title.split(token_delim)
            qry_words = []
            title_words = []
            for tokens in qry_tokens:
                token_seg = re.split(word_tag_delim + "|" + tag_tw_delim, tokens)
                if len(token_seg) != 3:
                    continue
                word = token_seg[0]
                qry_words.append(word)
            for tokens in title_tokens:
                token_seg = re.split(word_tag_delim + "|" + tag_tw_delim, tokens)
                if len(token_seg) != 3:
                    continue
                word = token_seg[0]
                title_words.append(word)
            if len(qry_words) <= 3 or len(title_words) <= 3:
                line_id += 1
                continue
            f.write(str(line_id) + "|" + "".join(qry_words) + "|" + "".join(title_words) + "\n")
            line_id += 1


def group_censor_prep(work_dir="./", file_ext=".txt", word_tag_delim="/",
                      tag_tw_delim=":", token_delim=" ", col_delim="|"):
    import os
    line_id = 0
    for file in os.listdir(work_dir):
        if file.endswith(file_ext):
            f_in = os.path.join(work_dir, file)
            with open(f_in, "r", encoding="utf-8") as f:
                lines = f.readlines()
            fn, ext = os.path.splitext(f_in)
            with open(fn + "_cdata" + ext, "w", encoding="utf-8") as f:
                for line in tqdm(lines, desc="Writing", leave=False, ascii=True):
                    line = line.rstrip()
                    seg_list = line.split(col_delim)
                    qry, title = seg_list[5], seg_list[7]
                    qry_tokens = qry.split(token_delim)
                    title_tokens = title.split(token_delim)
                    qry_words = []
                    title_words = []
                    for tokens in qry_tokens:
                        token_seg = re.split(word_tag_delim + "|" + tag_tw_delim, tokens)
                        if len(token_seg) != 3:
                            continue
                        word = token_seg[0]
                        qry_words.append(word)
                    for tokens in title_tokens:
                        token_seg = re.split(word_tag_delim + "|" + tag_tw_delim, tokens)
                        if len(token_seg) != 3:
                            continue
                        word = token_seg[0]
                        title_words.append(word)
                    if len(qry_words) <= 3 or len(title_words) <= 3:
                        line_id += 1
                        continue
                    f.write(str(line_id) + "|" + "".join(qry_words) + "|" + "".join(title_words) + "\n")
                    line_id += 1


def merge_files_to_one(in_fn_list, op_fn):
    switch_to_append = False
    for file in in_fn_list:
        if os.path.isfile(file):
            print("writing {} into {}...".format(file, op_fn))
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            code = "w" if not switch_to_append else "a"
            with open(op_fn, code, encoding="utf-8") as f:
                for line in lines:
                    line = line.rstrip()
                    if len(line) == 0: continue
                    f.write(line + "\n")
            switch_to_append = True


def read_wi_data(in_fn, ignore_header=False, col_delim="|", word_delim=" ",
                 wi_threshold=2, wi_limit=10):
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    rv = []
    for line in lines:
        line = line.rstrip()
        if len(line) == 0: continue
        segs = [w.split(word_delim) for w in line.split(col_delim)]
        assert len(segs) >= 2
        if wi_threshold is not None:
            if len(segs[1]) < wi_threshold: continue
        if wi_limit is not None:
            if len(segs[1]) > wi_limit: segs[1] = segs[1][:wi_limit]
        rv.append(segs)
    return rv


def stratified_sample_wi_bc(data_lists, valid_ratio=0.05, test_ratio=0.05, col_delim="|", word_delim=" "):
    uniques = set()
    pos_data, neg_data = [], []
    for d_list in data_lists:
        for segs in d_list:
            if segs[1][0] in segs[0]:
                continue
            key = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
            if key not in uniques:
                uniques.add(key)
                label = int(segs[2][0])
                if label == 0:
                    neg_data.append(segs)
                elif label == 1:
                    pos_data.append(segs)
                else:
                    assert False
    print("total pos len before: {}".format(len(pos_data)))
    print("total neg len before: {}".format(len(neg_data)))
    assert len(pos_data) > 0
    assert len(neg_data) > 0
    random.shuffle(pos_data)
    random.shuffle(neg_data)
    pos_data = pos_data[:min(len(pos_data), len(neg_data))]
    neg_data = neg_data[:min(len(pos_data), len(neg_data))]
    print("total pos len after: {}".format(len(pos_data)))
    print("total neg len after: {}".format(len(neg_data)))
    valid_pos_amount = int(len(pos_data) * valid_ratio)
    valid_neg_amount = int(len(neg_data) * valid_ratio)
    test_pos_amount = int(len(pos_data) * test_ratio)
    test_neg_amount = int(len(neg_data) * test_ratio)
    valid_pos = pos_data[:valid_pos_amount]
    valid_neg = neg_data[:valid_neg_amount]
    test_pos = pos_data[valid_pos_amount:valid_pos_amount+test_pos_amount]
    test_neg = neg_data[valid_neg_amount:valid_neg_amount+test_neg_amount]
    train_pos = pos_data[valid_pos_amount+test_pos_amount:]
    train_neg = neg_data[valid_neg_amount+test_neg_amount:]
    print("valid pos len {}".format(len(valid_pos)))
    print("valid neg len {}".format(len(valid_neg)))
    print("test pos len {}".format(len(test_pos)))
    print("test neg len {}".format(len(test_neg)))
    print("train pos len {}".format(len(train_pos)))
    print("train neg len {}".format(len(train_neg)))
    train_lines, valid_lines, test_lines = [], [], []
    for segs in train_pos:
        line = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
        train_lines.append(line)
    for segs in train_neg:
        line = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
        train_lines.append(line)
    for segs in valid_pos:
        line = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
        valid_lines.append(line)
    for segs in valid_neg:
        line = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
        valid_lines.append(line)
    for segs in test_pos:
        line = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
        test_lines.append(line)
    for segs in test_neg:
        line = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
        test_lines.append(line)
    random.shuffle(train_lines)
    random.shuffle(valid_lines)
    random.shuffle(test_lines)
    assert len(train_lines) > 0
    assert len(valid_lines) > 0
    assert len(test_lines) > 0
    return train_lines, valid_lines, test_lines


def write_wi_bc_data(data_lists,
                     train_fn="wi_bin_classify_train.txt",
                     valid_fn="wi_bin_classify_valid.txt",
                     test_fn="wi_bin_classify_test.txt",
                     valid_ratio=0.05, test_ratio=0.05, col_delim="|", word_delim=" "):
    print("Performing train, valid, test stratified sampling...")
    train_lines, valid_lines, test_lines = stratified_sample_wi_bc(data_lists,
                                                                   valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                                   col_delim=col_delim, word_delim=word_delim)
    print("Writing train, valid, test data...")
    with open(train_fn, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
    with open(valid_fn, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")
    with open(test_fn, "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line + "\n")
    print("{} lines wrote to {}".format(len(train_lines), train_fn))
    print("{} lines wrote to {}".format(len(valid_lines), valid_fn))
    print("{} lines wrote to {}".format(len(test_lines), test_fn))


def filter_contradicting_wi_bc_data(in_fn, op_fn, ignore_header=False, col_delim="|", word_delim=" "):
    print("Filtering contradicting data in {}".format(in_fn))
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    data = []
    pos, neg = set(), set()
    for line in lines:
        line = line.rstrip()
        if len(line) == 0: continue
        segs = [w.split(word_delim) for w in line.split(col_delim)]
        assert len(segs) == 3
        data.append(segs)
    for segs in data:
        key = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1])
        label = int(segs[2][0])
        if segs[1][0] in segs[0]:
            continue
        if label == 0:
            neg.add(key)
        elif label == 1:
            pos.add(key)
        else:
            assert False, "Unknown label {}".format(label)
    overlaps = pos.intersection(neg)
    if len(overlaps) == 0:
        print("No contradictions found in {}, no need to re-write".format(in_fn))
    else:
        print("{} contradictions found, re-writing to {}".format(len(overlaps), op_fn))
        wh = 0
        with open(op_fn, "w", encoding="utf-8") as f:
            for segs in data:
                if segs[1][0] in segs[0]:
                    continue
                key = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1])
                if key not in overlaps:
                    s = word_delim.join(segs[0]) + col_delim + word_delim.join(segs[1]) + col_delim + segs[2][0]
                    f.write(s + "\n")
                    wh += 1
        print("{} lines wrote to {}".format(wh, op_fn))


def read_wi_bin_classify_data(in_fn, ignore_header=False, col_delim="|", word_delim=" "):
    with open(in_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    rv = []
    for line in lines:
        line = line.rstrip()
        if len(line) == 0: continue
        segs = [w.split(word_delim) for w in line.split(col_delim)]
        assert len(segs) == 3
        rv.append(segs)
    return rv


def load_kw_pos(in_fn="postags_kws.txt"):
    d = os.path.dirname(os.path.abspath(__file__))
    with open(d + os.path.sep + in_fn, "r", encoding="utf-8") as f_:
        lines = f_.readlines()
    rv = set()
    for line in lines:
        line = line.rstrip().lower()
        if len(line) == 0: continue
        rv.add(line)
    return rv


def write_qry_pairs_from_qry_log_files(qry_log_files=["E:/files/query/qry_log_20181124.txt"]):
    cgl = filter_cg_list_by_repeated_clicks(qry_log_files, gen_qry_log_files, [4, 6, 20, 30], 2)
    # cgl = read_pre_seg_tw_data(qry_log_file, cache_file=qry_log_cache_file, light_weight=True)
    _, t2q = build_click_dicts(cgl)
    _w2c = build_w2c_from_seg_word_lists([qry.seg_list + title.seg_list for qry, title in cgl])
    all_pairs = select_representative_query(t2q, _w2c)
    assert len(all_pairs) > 0
    random.shuffle(all_pairs)
    test = all_pairs[:int(len(all_pairs)*0.05)]
    valid = all_pairs[int(len(all_pairs)*0.05):int(len(all_pairs)*0.1)]
    train = all_pairs[int(len(all_pairs)*0.1):]
    print("train_pairs len {}".format(len(train)))
    print("valid_pairs len {}".format(len(valid)))
    print("test_pairs len {}".format(len(test)))
    # lab_buckets = {}
    # for ind, pv in enumerate(all_pairs):
    #     if pv[1] not in lab_buckets: lab_buckets[pv[1]] = []
    #     lab_buckets[pv[1]].append(ind)
    # train, valid, test = bucket_sample_dev_test(all_pairs, lab_buckets)
    with open("test_pairs.txt", "w", encoding="utf-8") as f:
        for pv in test:
            f.write("|".join([" ".join(pv[0].seg_list), " ".join(pv[1].seg_list), ]) + "\n")
    with open("valid_pairs.txt", "w", encoding="utf-8") as f:
        for pv in valid:
            f.write("|".join([" ".join(pv[0].seg_list), " ".join(pv[1].seg_list), ]) + "\n")
    with open("train_pairs.txt", "w", encoding="utf-8") as f:
        for pv in train:
            f.write("|".join([" ".join(pv[0].seg_list), " ".join(pv[1].seg_list), ]) + "\n")


def build_wi_data(title2queries, w2c, allow_overlapping_words=True,
                  q_threshold_in_group=2, q_kws_threshold=2,
                  query_limit=50, wi_limit=10):
    rv = []
    uniques = set()
    # kws = load_kw_pos()
    # kws.add("v")
    exclude_title_words = {"百度"}
    for title, qs in tqdm(title2queries.items(), desc="building wi", ascii=True):
        q_list = [q for q, _ in qs.items()]
        if len(q_list) < q_threshold_in_group: continue
        if len(q_list) > query_limit:
            q_list = sorted(q_list, key=lambda q:len(q), reverse=False)[:query_limit]
        for q_tgt in q_list:
            q_cmp_sel_words = set()
            q_cmp_sel_word_scores = []
            q_tgt_words = set(q_tgt.seg_list)
            for q_cmp in q_list:
                if q_cmp == q_tgt and not allow_overlapping_words:
                    continue
                q_cmp_kws = list(set([w for w in q_cmp.keyword_seg_list])) # different extractor
                if len(q_cmp_kws) < q_kws_threshold: continue
                for word in q_cmp_kws:
                    if not allow_overlapping_words and word in q_tgt_words:
                        continue
                    if word not in q_cmp_sel_words and word in w2c and len(word) > 1:
                        q_cmp_sel_words.add(word)
                        q_cmp_sel_word_scores.append( (word, w2c[word]) )
                for word in title.keyword_seg_list:
                    if word in exclude_title_words: continue
                    if word not in q_cmp_sel_words and word in w2c and len(word) > 1:
                        q_cmp_sel_words.add(word)
                        q_cmp_sel_word_scores.append( (word, w2c[word]) )
            q_cmp_sel_word_scores = sorted(q_cmp_sel_word_scores, key=lambda t:(t[1], t[0]), reverse=True)[:wi_limit]
            feat_words = [t[0] for t in q_cmp_sel_word_scores]
            if len(feat_words) < q_kws_threshold: continue
            key = "|".join([" ".join(q_tgt.seg_list), " ".join(feat_words)])
            if key not in uniques:
                uniques.add(key)
                # rv.append([q_tgt.seg_list, feat_words, q_tgt.pos_list])
                rv.append([q_tgt.seg_list, feat_words, q_tgt.keyword_seg_list])
    return rv


def compress_click_graph(seg_list_gen, lines, thresholds=[], filter_threshold=2,
                         output_file="click_graph_compressed.txt"):
    counts = {}
    for pair in tqdm(seg_list_gen(lines), desc="Counting repeated clicks", total=len(lines)):
        qry = pair[0]
        title = pair[1]
        key = qry.raw_str + title.raw_str
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    sizes_by_thresholds = [
        (">={}".format(threshold), len([1 for k, v in counts.items() if v >= threshold]))
        for threshold in thresholds
    ]
    print(sizes_by_thresholds)
    write_head = 0
    wrote_keys = set()
    with open(output_file, "w", encoding="utf-8") as op_f:
        for pair in tqdm(seg_list_gen(lines), desc="Writing compressed data", total=len(lines)):
            qry = pair[0]
            title = pair[1]
            key = qry.raw_str + title.raw_str
            if counts[key] >= filter_threshold and key not in wrote_keys:
                op_f.write("|".join([" ".join(qry.seg_list), " ".join(title.seg_list),
                                     " ".join(qry.keyword_seg_list), " ".join(title.keyword_seg_list),
                                     str(counts[key])]) + "\n")
                write_head += 1
                wrote_keys.add(key)
    print("{} lines wrote to {}".format(write_head, output_file))


def filter_click_graph_by_repeated_clicks(seg_list_gen, lines, thresholds=[], filter_threshold=2):
    counts = {}
    for pair in tqdm(seg_list_gen(lines), desc="Counting repeated clicks", total=len(lines)):
        qry = pair[0]
        title = pair[1]
        key = qry.raw_str + title.raw_str
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    sizes_by_thresholds = [
        (">={}".format(threshold), len([1 for k, v in counts.items() if v >= threshold]))
        for threshold in thresholds
    ]
    print(sizes_by_thresholds)
    rv = []
    for pair in tqdm(seg_list_gen(lines), desc="filtering repeated clicks", total=len(lines)):
        qry = pair[0]
        title = pair[1]
        key = qry.raw_str + title.raw_str
        if counts[key] >= filter_threshold:
            rv.append(pair)
    return rv


def filter_cg_list_by_repeated_clicks(qry_log_files, seg_list_gen, thresholds=[], filter_threshold=2):
    import time
    counts = {}
    print("Counting repeated clicks")
    n = 1
    for gtr in seg_list_gen(qry_log_files):
        print("working on file {}".format(n))
        start = time.time()
        for t in gtr:
            qry = t[0]
            title = t[1]
            key = qry.raw_str + title.raw_str
            if key not in counts:
                counts[key] = 0
            counts[key] += 1
        elapsed = time.time() - start
        print("elapsed: {}".format(elapsed))
        n += 1
    sizes_by_thresholds = [
        (">={}".format(threshold), len([1 for k, v in counts.items() if v >= threshold]))
        for threshold in thresholds
    ]
    print(sizes_by_thresholds)
    rv = []
    n = 1
    for gtr in seg_list_gen(qry_log_files):
        print("filtering from file {}".format(n))
        for t in gtr:
            qry = t[0]
            title = t[1]
            key = qry.raw_str + title.raw_str
            if counts[key] >= filter_threshold:
                rv.append(t)
        n += 1
    return rv


def write_wi_pairs_from_qry_logs(qry_log_files=["E:/files/query/qry_log_20181124.txt"],
                                 output_file_header="wi", append=False):
    cg_list = filter_cg_list_by_repeated_clicks(qry_log_files, gen_qry_log_files, [2,3,5,10], 2)
    q2t, t2q = build_click_dicts(cg_list)
    q2t, t2q = filter_contradicting_qrys(q2t, t2q)
    w2c = build_w2c_from_seg_word_lists([t[0].seg_list for t in cg_list])
    data = build_wi_data(t2q, w2c)
    # data = clean_wi_data(data)
    assert len(data) > 0
    random.shuffle(data)
    print("built wi data len {}".format(len(data)))
    d_test = data[:int(len(data) * 0.05)]
    d_valid = data[int(len(data) * 0.05):int(len(data) * 0.1)]
    d_train = data[int(len(data) * 0.1):]
    code = "a" if append else "w"
    with open(output_file_header + "_train.txt", code, encoding="utf-8") as f:
        for row in d_train:
            f.write("|".join([" ".join(row[i]) for i in range(len(row))])+"\n")
    with open(output_file_header + "_valid.txt", code, encoding="utf-8") as f:
        for row in d_valid:
            f.write("|".join([" ".join(row[i]) for i in range(len(row))])+"\n")
    with open(output_file_header + "_test.txt", code, encoding="utf-8") as f:
        for row in d_test:
            f.write("|".join([" ".join(row[i]) for i in range(len(row))])+"\n")


def fast_build_wi_bin_classify_data(title2queries, query_limit=20, q_threshold_in_group=2,
                                    q_kws_threshold=2):
    """
    No contradictions check
    """
    rv = []
    all_words = set()
    added_pos_insts = set()
    added_neg_insts = set()
    for title, qs in tqdm(title2queries.items(), desc="Preparing word set", ascii=True):
        q_list = [q for q, _ in qs.items()]
        if len(q_list) < q_threshold_in_group: continue
        if len(q_list) > query_limit:
            q_list = sorted(q_list, key=lambda q: len(q), reverse=False)[:query_limit]
        for q in q_list:
            for word in q.keyword_seg_list:
                all_words.add(word)
        for word in title.keyword_seg_list:
            all_words.add(word)
    all_words_list = list(all_words)
    for title, qs in tqdm(title2queries.items(), desc="Building data", ascii=True):
        q_list = [q for q, _ in qs.items()]
        if len(q_list) < q_threshold_in_group: continue
        if len(q_list) > query_limit:
            q_list = sorted(q_list, key=lambda q: len(q), reverse=False)[:query_limit]
        for q_tgt in q_list:
            q_tgt_words = set(q_tgt.seg_list)
            q_cmp_words = set()
            for q_cmp in q_list:
                if q_cmp == q_tgt: continue
                for word in q_cmp.keyword_seg_list:
                    if len(word) > 1:
                        q_cmp_words.add(word)
            q_cmp_kws = list(q_cmp_words-q_tgt_words)
            if len(q_cmp_kws) < q_kws_threshold: continue
            for pos_word in q_cmp_kws:
                if pos_word in q_tgt_words: continue
                pos_key =str(q_tgt) + "|" + pos_word
                if pos_key in added_pos_insts:
                    continue
                neg_word = random.choice(all_words_list)
                neg_key = str(q_tgt) + "|" + neg_word
                watchdog = 0
                while neg_word in q_tgt_words or neg_word in q_cmp_words or neg_key in added_neg_insts or neg_key in added_pos_insts:
                    if watchdog >= 1e7: break
                    neg_word = random.choice(all_words_list)
                    neg_key = str(q_tgt) + "|" + neg_word
                    watchdog += 1
                assert watchdog < 1e7, "watchdog triggered"
                added_pos_insts.add(pos_key)
                added_neg_insts.add(neg_key)
                rv.append([q_tgt.seg_list, [pos_word], [str(1)]])
                rv.append([q_tgt.seg_list, [neg_word], [str(0)]])
    return rv


def build_wi_bin_classify_data(title2queries, w2c, w2v, query_limit=20, wi_limit=10,
                               randomly_choose=True):
    rv = []
    title_to_query_words = {}
    most_similar_titles = {}
    unique_pos = set()
    for title, qs in tqdm(title2queries.items(), desc="Preparing word set", ascii=True):
        q_list = [q for q, _ in qs.items()]
        if len(q_list) > query_limit:
            q_list = sorted(q_list, key=lambda q: len(q), reverse=False)[:query_limit]
        if title not in title_to_query_words:
            word_set = set()
            for q in q_list:
                for word in q.keyword_seg_list:
                    word_set.add(word)
                    unique_pos.add(str(q) + "|" + word)
            title_to_query_words[title] = word_set

    if randomly_choose:
        rand_titles = [title for title, _ in title2queries.items()]
        for title, qs in tqdm(title2queries.items(), desc="Randomly choosing negative title", ascii=True):
            cand = random.choice(rand_titles)
            watchdog = 0
            while cand == title and watchdog < 1e7:
                cand = random.choice(rand_titles)
                watchdog += 1
            assert watchdog < 1e7, "watchdog triggered"
            most_similar_titles[title] = cand
    else:
        keyword_to_titles = {}
        for title, _ in tqdm(title2queries.items(), desc="Preparing keyword to titles", ascii=True):
            for w in title.keyword_seg_list:
                if w not in keyword_to_titles: keyword_to_titles[w] = set()
                keyword_to_titles[w].add(title)
        for title, _ in tqdm(title2queries.items(), desc="Preparing query pairs", ascii=True):
            if title not in title_to_query_words: continue
            overlap_scores = {}
            for w in title.keyword_seg_list:
                watchdog = 0
                if w not in keyword_to_titles: continue
                overlap_titles = list(keyword_to_titles[w])
                for title_comp in overlap_titles:
                    if title_comp not in overlap_scores: overlap_scores[title_comp] = 0
                    overlap_scores[title_comp] += 1
                    watchdog += 1
                    if watchdog > 1e2: break
            if len(overlap_scores) == 0: continue
            most_similar_title = None
            best_score = None
            watchdog = 0
            for k, v in overlap_scores.items():
                if best_score is None or v >  best_score:
                    best_score = v
                    most_similar_title = k
                    watchdog += 1
                    if watchdog > 1e4: break
            if most_similar_title is None: continue
            most_similar_titles[title] = most_similar_title
    for title, qs in tqdm(title2queries.items(), desc="Building data", ascii=True):
        title_neg = most_similar_titles[title]
        words_neg = list(title_to_query_words[title_neg])
        if len(words_neg) == 0: continue
        q_list = [q for q, _ in qs.items()]
        if len(q_list) > query_limit:
            q_list = sorted(q_list, key=lambda q:len(q), reverse=False)[:query_limit]
        if len(q_list) <= 1: continue
        for q_tgt in q_list:
            q_cmp_sel_words = set()
            q_cmp_sel_word_scores = []
            q_tgt_words = set(q_tgt.seg_list)
            for q_cmp in q_list:
                if q_cmp == q_tgt: continue
                q_cmp_kws = [w for w in q_cmp.keyword_seg_list]
                for word in q_cmp_kws:
                    if word not in q_cmp_sel_words and word in w2c and word not in q_tgt_words and len(word) > 1:
                        q_cmp_sel_words.add(word)
                        q_cmp_sel_word_scores.append( (word, w2c[word]) )
                for word in title.keyword_seg_list:
                    if word not in q_cmp_sel_words and word in w2c and word not in q_tgt_words and len(word) > 1:
                        q_cmp_sel_words.add(word)
                        q_cmp_sel_word_scores.append( (word, w2c[word]) )
            q_cmp_sel_word_scores = sorted(q_cmp_sel_word_scores, key=lambda t:(t[1], t[0]), reverse=True)[:wi_limit]
            feat_words = [t[0] for t in q_cmp_sel_word_scores]
            if len(feat_words) < 2: continue

            for w in feat_words:
                if w in w2v:
                    w_vec = w2v[w]
                    w_sims = []
                    for w_comp in words_neg:
                        if w_comp == w: continue
                        if w_comp not in w2v:
                            w_sims.append([w_comp, -1.0])
                            continue
                        w_sims.append([w_comp, cosine_sim_np(w_vec, w2v[w_comp])])
                    if len(w_sims) == 0: continue
                    w_sims = sorted(w_sims, key=lambda t:t[1], reverse=True)
                    w_neg = w_sims[0][0]
                else:
                    w_neg = random.choice(words_neg)
                rv.append([q_tgt.seg_list, w, str(1)])
                rv.append([q_tgt.seg_list, w_neg, str(0)])
    return rv


def build_bin_classify_data_from_wi(data_list, negative_words_set):
    rv = []
    negative_words = list(negative_words_set)
    for row in tqdm(data_list, desc="Build wi bc data", ascii=True):
        pos_words_set = set(row[1])
        if len(pos_words_set) == 0: continue
        for pos_word in row[1]:
            neg_word = random.choice(negative_words)
            if len(pos_word) == 0 or len(neg_word) == 0: continue
            watchdog = 0
            while neg_word in pos_words_set:
                neg_word = random.choice(negative_words)
                watchdog += 1
                if watchdog > 1e4: break
            if watchdog >= 1e4: continue
            rv.append([row[0], [pos_word], [str(1)]])
            rv.append([row[0], [neg_word], [str(0)]])
    assert len(rv) > 0
    return rv


def write_wi_bin_classify_data_from_qry_logs(qry_log_files=["E:/files/query/qry_log_20181124.txt"],
                                             op_fn = "wi_bin_classify_data.txt"):
    import gensim
    cg_list = filter_cg_list_by_repeated_clicks(qry_log_files, gen_qry_log_files, [7], 2)
    # cg_list = read_pre_seg_tw_data(qry_log_file, cache_file=qry_log_cache_file, read_lines_limit=read_lines_limit,
    #                                light_weight=True)
    _, t2q = build_click_dicts(cg_list)
    # w2c = build_w2c_from_seg_word_lists([qry.seg_list for qry, _ in cg_list])
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(word_vec_file, binary=False)
    # data = build_wi_bin_classify_data(t2q, w2c, w2v)
    data = fast_build_wi_bin_classify_data(t2q)
    assert len(data) > 0
    random.shuffle(data)
    print("Writing wi bin classify data len {}".format(len(data)))
    with open(op_fn, "w", encoding="utf-8") as f:
        for row in data:
            f.write("|".join([" ".join(row[i]) for i in range(len(row))])+"\n")


def filter_contradicting_qrys(q2t, t2q, topk=1):
    print("keeping only top {} title(s) for every query!".format(topk))
    filtered_q2t, filtered_t2q = {}, {}
    contradicting_qrys = set()
    for qry, titles in tqdm(q2t.items(), desc="Filtering qry title dicts", ascii=True):
        original_titles = [(t,c) for t,c in titles.items()]
        if len(titles) <= topk:
            selected_titles = original_titles
        else:
            sorted_titles = sorted(original_titles, key=lambda t:t[1], reverse=True)
            selected_titles = sorted_titles[:topk]
            contradicting_qrys.add(qry)
        for t, c in selected_titles:
            if qry not in filtered_q2t: filtered_q2t[qry] = {}
            filtered_q2t[qry][t] = c
            if t not in filtered_t2q: filtered_t2q[t] = {}
            filtered_t2q[t][qry] = c
    print("{} contradicting qrys found".format(len(contradicting_qrys)))
    print("title2qrys len after filtering {}".format(len(filtered_t2q)))
    print("qry2titles len after filtering {}".format(len(filtered_q2t)))
    return filtered_q2t, filtered_t2q


def clean_wi_data(data, rm_n_high_freq_words=10, rm_n_low_freq_words=10):
    rv = []
    # remove words that are too frequent or too rare
    exclude_words = set()
    # word_to_qry_count = {}
    # for segs in tqdm(data, desc="Cleaning", ascii=True):
    #     for word in segs[1]:
    #         if word not in word_to_qry_count:
    #             word_to_qry_count[word] = set()
    #         key = " ".join(segs[0])
    #         word_to_qry_count[word].add(key)
    # sorted_words = sorted([ (w, qs) for w, qs in word_to_qry_count.items()], key=lambda t: len(t[1]), reverse=True)
    # for w, _ in sorted_words[:rm_n_high_freq_words]:
    #     exclude_words.add(w)
    # for w, _ in sorted_words[(-1 * rm_n_low_freq_words):]:
    #     exclude_words.add(w)
    # for segs in data:
    #     new_words = [w for w in segs[1] if w not in exclude_words]
    #     if len(new_words) < 1: continue
    #     rv.append([segs[0], new_words])

    # check for duplicate qrys
    unqiue_qrys = {}
    qry_to_data = {}
    for segs in tqdm(data, desc="Cleaning", ascii=True):
        key = " ".join(segs[0])
        if key not in unqiue_qrys:
            unqiue_qrys[key] = 0
            qry_to_data[key] = set()
        unqiue_qrys[key] += 1
        qry_to_data[key] = set(segs[1]).union(qry_to_data[key])
    dup_qrys = {k:v for k,v in unqiue_qrys.items() if v > 1}
    print("{} duplciate qrys in wi data".format(len(dup_qrys)))
    rv = [[k.split(" "),list(v)] for k,v in qry_to_data.items()]
    print("{} unique wi data pairs returned".format(len(rv)))
    return rv


def merge_wi_qrw_data(wi_data_file, qrw_data_file, col_delim="|"):
    def _prune_spaces(in_str):
        return re.sub(r" +", r" ", in_str).strip()
    with open(wi_data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    wi_data = [line.rstrip().split(col_delim) for line in lines if len(line) > 0]
    with open(qrw_data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    qrw_data = [line.rstrip().split(col_delim) for line in lines if len(line) > 0]
    rv = []
    wi_keys = {}
    for row in tqdm(wi_data, desc="Building wi keys", ascii=True):
        key = row[0]
        wi_keys[key] = row[1]
    print("wi keys len {}".format(len(wi_keys)))
    write_head = 0
    for row in tqdm(qrw_data, desc="Merging wi qrw data", ascii=True):
        key = row[0]
        if key in wi_keys:
            write_head += 1
            wi_row = wi_keys[key]
            rv.append([_prune_spaces(row[0]), _prune_spaces(row[1]), _prune_spaces(wi_row)])
    print("{} rows of data after merging".format(write_head))
    return rv


def filter_seg_list_duplicates(seg_list):
    keys = set()
    duplicates = 0
    rv = []
    for row in seg_list:
        # key = "|".join([" ".join(r) for r in row])
        key = "|".join([" ".join(row[0]), " ".join(row[2])])
        if key in keys:
            duplicates += 1
        else:
            keys.add(key)
            rv.append(row)
    print("{} duplicates found".format(duplicates))
    return rv


def filter_seg_list_by_rules(seg_list):
    rv = []
    filtered_out_count = 0
    for row in seg_list:
        if len(row[0]) > 20 or len(row[0]) == 0:
            filtered_out_count += 1
            continue
        if len(row[1]) > 20 or len(row[0]) == 0:
            filtered_out_count += 1
            continue
        # if "".join(row[1]) in "".join(row[0]):
        #     filtered_out_count += 1
        #     continue
        rv.append(row)
    print("{} instances filtered out by rules".format(filtered_out_count))
    return rv

def fix_qg_wi_data(in_fn, op_fn):
    rv = []
    data = read_col_word_delim_data(in_fn)
    for row in data:
        qrw_wi = row[3]
        tgt = row[2]
        title = row[1]
        qry = row[0]
        wi = qrw_wi[len(qry):]
        qg_wi = title + wi
        new_row = [qry, title, tgt, qrw_wi, qg_wi]
        rv.append(new_row)
    with open(op_fn, "w", encoding="utf-8") as f:
        for row in rv:
            f.write("|".join([" ".join(wl) for wl in row]) + "\n")


if __name__ == "__main__":
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_2 = "E:/files/query"
    q_log_files = [
        # dir_path + "/qry_log_20181115.txt",
        # dir_path + "/qry_log_20181124.txt",
        # dir_path + "/qry_log_20181126.txt",
        # dir_path + "/qry_log_20181128.txt",
        # dir_path + "/qry_log_20181130.txt",
        # dir_path + "/qry_log_20181202.txt",
        # dir_path + "/qry_log_20181207.txt",
        # dir_path + "/qry_log_20181209.txt",

        dir_path_2 + "/qry_log_20181115.txt",
        dir_path_2 + "/qry_log_20181124.txt",
        dir_path_2 + "/qry_log_20181126.txt",
        dir_path_2 + "/qry_log_20181128.txt",
        dir_path_2 + "/qry_log_20181130.txt",
        dir_path_2 + "/qry_log_20181202.txt",
        dir_path_2 + "/qry_log_20181207.txt",
        dir_path_2 + "/qry_log_20181209.txt",
    ]
    with open("click_graph.txt", "r", encoding="utf-8") as f:
        ls = f.readlines()
    compress_click_graph(click_graph_generator, ls,
                         thresholds=[2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100, 200, 500],
                         filter_threshold=2)
    build_wi_qrw_data_from_click_graph("click_graph_compressed.txt")
    # write_simplified_click_graph(q_log_files, "click_graph_small.txt")
    # from utils.lang_utils import load_gensim_word_vec
    # pre_built_w2v = load_gensim_word_vec("../libs/Tencent_AILab_ChineseEmbedding.txt",
    #                                      cache_file="../cache/tencent_ailab_w2v.pkl")
    # merge_files_to_one([
    #     "qrw_train.txt",
    #     "qrw_valid.txt",
    #     "qrw_test.txt",
    # ], "qrw_data.txt")
    # wq_data = read_qrw_pairs_data("qrw_data.txt")
    # wq_data = filter_seg_list_duplicates(wq_data)
    # wq_data = filter_seg_list_by_rules(wq_data)
    # random.shuffle(wq_data)
    # # wq_data = wq_data[:int(0.5*len(wq_data))]
    # test = wq_data[:int(len(wq_data) * 0.05)]
    # valid = wq_data[int(len(wq_data) * 0.05):int(len(wq_data) * 0.1)]
    # train = wq_data[int(len(wq_data) * 0.1):]
    # print("train_pairs len {}".format(len(train)))
    # print("valid_pairs len {}".format(len(valid)))
    # print("test_pairs len {}".format(len(test)))
    # with open("qrw_test.txt", "w", encoding="utf-8") as f:
    #     for pv in test:
    #         f.write("|".join([" ".join(wl) for wl in pv]) + "\n")
    # with open("qrw_valid.txt", "w", encoding="utf-8") as f:
    #     for pv in valid:
    #         f.write("|".join([" ".join(wl) for wl in pv]) + "\n")
    # with open("qrw_train.txt", "w", encoding="utf-8") as f:
    #     for pv in train:
    #         f.write("|".join([" ".join(wl) for wl in pv]) + "\n")
    #
    # with open("wi_test.txt", "w", encoding="utf-8") as f:
    #     for pv in test:
    #         f.write("|".join([" ".join(pv[0]), " ".join(pv[2])]) + "\n")
    # with open("wi_valid.txt", "w", encoding="utf-8") as f:
    #     for pv in valid:
    #         f.write("|".join([" ".join(pv[0]), " ".join(pv[2])]) + "\n")
    # with open("wi_train.txt", "w", encoding="utf-8") as f:
    #     for pv in train:
    #         f.write("|".join([" ".join(pv[0]), " ".join(pv[2])]) + "\n")

    # wi_train = read_wi_data("wi_train.txt")
    # wi_valid = read_wi_data("wi_valid.txt")
    # wi_test = read_wi_data("wi_test.txt")
    #
    # neg_words = set()
    # for _row in wi_train:
    #     for w in _row[0] + _row[1]:
    #         if w in pre_built_w2v:
    #             neg_words.add(w)
    #
    # wi_bc_train = build_bin_classify_data_from_wi(wi_train, neg_words)
    # wi_bc_valid = build_bin_classify_data_from_wi(wi_valid, neg_words)
    # wi_bc_test = build_bin_classify_data_from_wi(wi_test, neg_words)
    # random.shuffle(wi_bc_train)
    # random.shuffle(wi_bc_valid)
    # random.shuffle(wi_bc_test)
    # with open("wi_bin_classify_train.txt", "w", encoding="utf-8") as f:
    #     for pv in wi_bc_train:
    #         f.write("|".join([" ".join(wl) for wl in pv]) + "\n")
    # with open("wi_bin_classify_valid.txt", "w", encoding="utf-8") as f:
    #     for pv in wi_bc_valid:
    #         f.write("|".join([" ".join(wl) for wl in pv]) + "\n")
    # with open("wi_bin_classify_test.txt", "w", encoding="utf-8") as f:
    #     for pv in wi_bc_test:
    #         f.write("|".join([" ".join(wl) for wl in pv]) + "\n")
    # filter_contradicting_wi_bc_data("wi_bin_classify_train.txt", "wi_bin_classify_train.txt")
    # filter_contradicting_wi_bc_data("wi_bin_classify_valid.txt", "wi_bin_classify_valid.txt")
    # filter_contradicting_wi_bc_data("wi_bin_classify_test.txt", "wi_bin_classify_test.txt")
    #
    # merge_files_to_one([
    #     "wi_bin_classify_train.txt",
    #     "wi_bin_classify_valid.txt",
    #     "wi_bin_classify_test.txt",
    # ], "wi_bin_classify_data.txt")
    # filter_contradicting_wi_bc_data("wi_bin_classify_data.txt", "wi_bin_classify_data.txt")


    # write_wi_pairs_from_qry_logs(qry_log_files=q_log_files, output_file_header="wi")

    # write_qry_pairs_from_qry_log_files(qry_log_files=q_log_files)
    #
    # write_wi_bin_classify_data_from_qry_logs(qry_log_files=q_log_files)
    # filter_contradicting_wi_bc_data("wi_bin_classify_data.txt", "wi_bin_classify_data.txt")
    # data_all = read_wi_bin_classify_data("wi_bin_classify_data.txt")
    # write_wi_bc_data([data_all])
    # assert len(data_all) > 0


    # merge_files_to_one([
    #     "wi_train.txt",
    #     "wi_valid.txt",
    #     "wi_test.txt",
    # ], "wi_data.txt")
    # wi_data = read_wi_data("wi_data.txt")
    # wi_data = clean_wi_data(wi_data)
    # assert len(wi_data) > 0
    # random.shuffle(wi_data)
    # print("write wi data len {}".format(len(wi_data)))
    # d_test = wi_data[:int(len(wi_data) * 0.05)]
    # d_valid = wi_data[int(len(wi_data) * 0.05):int(len(wi_data) * 0.1)]
    # d_train = wi_data[int(len(wi_data) * 0.1):]
    # with open("wi_train.txt", "w", encoding="utf-8") as f:
    #     for row in d_train:
    #         f.write("|".join([" ".join(row[i]) for i in range(len(row))]) + "\n")
    # with open("wi_valid.txt", "w", encoding="utf-8") as f:
    #     for row in d_valid:
    #         f.write("|".join([" ".join(row[i]) for i in range(len(row))]) + "\n")
    # with open("wi_test.txt", "w", encoding="utf-8") as f:
    #     for row in d_test:
    #         f.write("|".join([" ".join(row[i]) for i in range(len(row))]) + "\n")


    # merge_files_to_one([
    #     "train_pairs.txt",
    #     "valid_pairs.txt",
    #     "test_pairs.txt",
    # ], "qrw_data_tmp.txt")
    # merge_files_to_one([
    #     "wi_train.txt",
    #     "wi_valid.txt",
    #     "wi_test.txt",
    # ], "wi_data_tmp.txt")
    # wq_data = merge_wi_qrw_data("wi_data_tmp.txt", "qrw_data_tmp.txt")
    # random.shuffle(wq_data)
    # test = wq_data[:int(len(wq_data) * 0.05)]
    # valid = wq_data[int(len(wq_data) * 0.05):int(len(wq_data) * 0.1)]
    # train = wq_data[int(len(wq_data) * 0.1):]
    # print("train_pairs len {}".format(len(train)))
    # print("valid_pairs len {}".format(len(valid)))
    # print("test_pairs len {}".format(len(test)))
    # with open("qrw_test.txt", "w", encoding="utf-8") as f:
    #     for pv in test:
    #         f.write("|".join(pv) + "\n")
    # with open("qrw_valid.txt", "w", encoding="utf-8") as f:
    #     for pv in valid:
    #         f.write("|".join(pv) + "\n")
    # with open("qrw_train.txt", "w", encoding="utf-8") as f:
    #     for pv in train:
    #         f.write("|".join(pv) + "\n")
    # fix_qg_wi_data("qrw_wi_dv_train.txt", "qrw_qg_wi_dv_train.txt")
    # fix_qg_wi_data("qrw_wi_dv_valid.txt", "qrw_qg_wi_dv_valid.txt")
    # fix_qg_wi_data("qrw_wi_dv_test.txt", "qrw_qg_wi_dv_test.txt")
    print("done")
