from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import unicodedata


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def preprocess_eng_sent_to_seg_list(sent, lemmatizer=WordNetLemmatizer(),
                                    remove_stop_words=False, lemmantize=False,
                                    remove_chars=r"[^a-zA-Z0-9?,]+", number_replacement="#"):
    sent = unicodeToAscii(sent.lower().rstrip().strip())
    sent = sent.replace("'", "")
    sent = re.sub(r"([.,!?])", r" \1", sent)
    sent = re.sub(remove_chars, r" ", sent)
    sent = re.sub(r"\s+", r" ", sent).strip()
    sent_list = word_tokenize(sent)
    if remove_stop_words:
        eng_stops = set(stopwords.words('english'))
        sent_list = [w for w in sent_list if w not in eng_stops and len(w) > 1]
    if lemmantize:
        sent_list = [lemmatizer.lemmatize(w) for w in sent_list]
    sent_list = [number_replacement if is_number(w) else w for w in sent_list]
    return sent_list


def preprocess_chinese_raw_str(in_str, collapse_numbers=False):
    from zhon.hanzi import punctuation
    in_str = in_str.lower()
    in_str = re.sub(u'[^\u4E00-\u9FA5\u0030-\u0039A-Za-z，。：；:;,?!？！\.]', "", in_str)
    # in_str = re.sub(u"[%s]+" % punctuation, "", in_str)
    in_str = re.sub("。+", "。", in_str)
    in_str = re.sub("，+", "，", in_str)
    in_str = re.sub("[\.,：；:;、]+", "，", in_str)
    in_str = re.sub("[！？?!]+", "。", in_str)
    if collapse_numbers:
        in_str = re.sub(u'[\u0030-\u0039]+', "#", in_str)
        in_str = re.sub(u'#\.+#', "#", in_str)
    return in_str


def preprocess_english_raw_str(in_str, collapse_numbers=False):
    in_str = in_str.lower()
    in_str = re.sub(u'[^A-Za-z0-9:;,?!\.\s\-]', "", in_str)
    in_str = re.sub("\.com", "", in_str)
    in_str = re.sub("\.+", ". ", in_str)
    in_str = re.sub(",+", ", ", in_str)
    in_str = re.sub("[:;]+", ", ", in_str)
    in_str = re.sub("[?!]+", ". ", in_str)
    if collapse_numbers:
        in_str = re.sub(u'[0-9]+', "#", in_str)
        in_str = re.sub(u'#\.+#', "#", in_str)
    in_str = re.sub(r"\-+", r" ", in_str).strip()
    in_str = re.sub(r" +", r" ", in_str).strip()
    return in_str


def no_chinese_char_found(in_str):
    return re.sub(u'[^\u4E00-\u9FA5]', '', in_str) == ''


def traditional_to_simplified_chinese(in_seg_lists):
    from opencc import OpenCC
    cc = OpenCC('t2s')
    rv = []
    for seg_list in in_seg_lists:
        tmp = []
        for w in seg_list:
            wt = cc.convert(w)
            tmp.append(wt)
        rv.append(tmp)
    return rv


if __name__ == "__main__":
    print("done")
