import nltk
import math
from nltk.translate.bleu_score import SmoothingFunction
from constants import OOV_TOKEN


def post_evaluate_test_results_file(f_name="test_results.txt", col_delim="|", word_delim=" ",
                                    ignore_header=True, oov_token=OOV_TOKEN):
    with open(f_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:] if ignore_header else lines
    lsplit = [line.rstrip().strip().split(col_delim) for line in lines]
    gen_count = 0
    n_oov_tokens= 0
    n_tokens = 0
    rv = {}
    corpus_preds = []
    corpus_truth = []
    counted_results = set()
    for val_list in lsplit:
        if len(val_list) != 3: continue
        gen = val_list[1]
        truth = val_list[2]
        gen_count += 1
        truth_seg = [w for w in truth.split(word_delim) if len(w) > 0]
        gen_seg = [w for w in gen.split(word_delim) if len(w) > 0]
        # key = " ".join(truth_seg) + "|" + " ".join(gen_seg)
        # if key in counted_results: continue
        # counted_results.add(key)
        n_oov_tokens += sum([1 for w in gen.split(word_delim) if w == oov_token])
        n_tokens += sum([1 for w in gen.split(word_delim) if len(w) > 0])
        corpus_preds.append(gen_seg)
        corpus_truth.append([truth_seg])
        multi_eval(gen_seg, truth_seg, result_dict=rv)
    if gen_count == 0: return rv
    for k, v in rv.items():
        rv[k] = v / gen_count
    rv["corpus_bleu_4"] = bleu_c(corpus_preds, corpus_truth)
    rv["percent_oov_in_output"] = n_oov_tokens/n_tokens if n_tokens > 0 else -1.0
    return rv


def corpus_eval(pred_lists, truth_lists):
    """
    :param pred_lists: shape like [ [pred_word1, pred_word2...], [pred_word1, pred_word2...] ]
    :param truth_lists: shape like [ [[truth_word1, truth_word2...]], [[truth_word1, truth_word2...]] ]
    :param result_dict:
    """
    assert len(pred_lists) > 0, "pred_lists cannot be empty"
    assert len(pred_lists) == len(truth_lists), "One prediction must correspond to one truth"
    rv = {
        "bleu_1":0,
        "bleu_2":0,
        "bleu_3":0,
        "bleu_4":0,
        "rouge_1":0,
        "rouge_2":0,
        "rouge_L":0,
        "em":0,
        "acc":0,
    }
    for i, pred_seg in enumerate(pred_lists):
        truth_seg = truth_lists[i][0] # TODO: support multiple truth?
        rv["bleu_1"] += bleu_1(pred_seg, truth_seg)
        rv["bleu_2"] += bleu_2(pred_seg, truth_seg)
        rv["bleu_3"] += bleu_3(pred_seg, truth_seg)
        rv["bleu_4"] += bleu_4(pred_seg, truth_seg)
        rv["rouge_1"] += rouge_1(pred_seg, truth_seg)
        rv["rouge_2"] += rouge_2(pred_seg, truth_seg)
        rv["rouge_L"] += rouge_L(pred_seg, truth_seg)
        rv["em"] += em(pred_seg, truth_seg)
        rv["acc"] += em(pred_seg, truth_seg)
    for k, v in rv.items():
        rv[k] = v / len(pred_lists) # macro average
    rv["corpus_bleu_4"] = bleu_c(pred_lists, truth_lists)
    return rv


def multi_eval(gen_seg, truth_seg, result_dict={}):
    if "em" not in result_dict: result_dict["em"] = 0
    result_dict["em"] += em(gen_seg, truth_seg)
    if "acc" not in result_dict: result_dict["acc"] = 0
    result_dict["acc"] += em(gen_seg, truth_seg)
    if "bleu_1" not in result_dict: result_dict["bleu_1"] = 0
    result_dict["bleu_1"] += bleu_1(gen_seg, truth_seg)
    if "bleu_2" not in result_dict: result_dict["bleu_2"] = 0
    result_dict["bleu_2"] += bleu_2(gen_seg, truth_seg)
    if "bleu_3" not in result_dict: result_dict["bleu_3"] = 0
    result_dict["bleu_3"] += bleu_3(gen_seg, truth_seg)
    if "bleu_4" not in result_dict: result_dict["bleu_4"] = 0
    result_dict["bleu_4"] += bleu_4(gen_seg, truth_seg)
    if "rouge_1" not in result_dict: result_dict["rouge_1"] = 0
    result_dict["rouge_1"] += rouge_1(gen_seg, truth_seg)
    if "rouge_2" not in result_dict: result_dict["rouge_2"] = 0
    result_dict["rouge_2"] += rouge_2(gen_seg, truth_seg)
    if "rouge_L" not in result_dict: result_dict["rouge_L"] = 0
    result_dict["rouge_L"] += rouge_L(gen_seg, truth_seg)
    return result_dict


def perplexity(probs, truth_idx):
    """
    Not working yet
    """
    n = truth_idx.shape[0]
    val = 0
    for i in range(n):
        wi = truth_idx[i].item()
        prob = probs[i, wi].item()
        val += math.log(prob, 2)
    return math.log(2, -1 / n * val)


def bleu_c(predict, truth, n=4):
    smoother = SmoothingFunction()
    if n == 1:
        return 100 * nltk.translate.bleu_score.corpus_bleu(truth, predict, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
    if n == 2:
        return 100 * nltk.translate.bleu_score.corpus_bleu(truth, predict, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother.method1)
    if n == 3:
        return 100 * nltk.translate.bleu_score.corpus_bleu(truth, predict, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother.method1)
    if n == 4:
        return 100 * nltk.translate.bleu_score.corpus_bleu(truth, predict, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method1)
    print("n must be 1, 2, 3 or 4!")
    return None


def em(pred_seg_list, truth_seg_list):
    if " ".join(pred_seg_list) == " ".join(truth_seg_list):
        return 1
    else:
        return 0


def bleu_1(pred_seg_list, truth_seg_list):
    return 100 * _bleu_n(pred_seg_list, truth_seg_list, weights=(1,0,0,0))


def bleu_2(pred_seg_list, truth_seg_list):
    return 100 * _bleu_n(pred_seg_list, truth_seg_list, weights=(0.5,0.5,0,0)) # cumulative, not individual


def bleu_3(pred_seg_list, truth_seg_list):
    return 100 * _bleu_n(pred_seg_list, truth_seg_list, weights=(0.33,0.33,0.33,0))


def bleu_4(pred_seg_list, truth_seg_list):
    return 100 * _bleu_n(pred_seg_list, truth_seg_list, weights=(0.25,0.25,0.25,0.25))


def _bleu_n(pred_seg_list, truth_seg_list, weights):
    smoother = SmoothingFunction()
    score = nltk.translate.bleu_score.sentence_bleu([truth_seg_list], pred_seg_list,
                                                    weights=weights, smoothing_function=smoother.method1)
    return score


def rouge_L(pred_seg_list, truth_seg_list):
    rouge_score_map = rouge([" ".join(pred_seg_list)], [" ".join(truth_seg_list)])
    return 100 * rouge_score_map["rouge_l/f_score"]


def rouge_1(pred_seg_list, truth_seg_list):
    rouge_score_map = rouge([" ".join(pred_seg_list)], [" ".join(truth_seg_list)])
    return 100 * rouge_score_map["rouge_1/f_score"]


def rouge_2(pred_seg_list, truth_seg_list):
    rouge_score_map = rouge([" ".join(pred_seg_list)], [" ".join(truth_seg_list)])
    return 100 * rouge_score_map["rouge_2/f_score"]


"""
ROUGE metric implementation.

Copy from tf_seq2seq/seq2seq/metrics/rouge.py.
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/squad_dev/sumy/evaluation/rouge.py.
"""

import itertools
import numpy as np

#pylint: disable=C0103


def _get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _recon_lcs(x, y):
  """
  Returns the Longest Subsequence between x and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns:
    sequence: LCS of x and y
  """
  i, j = len(x), len(y)
  table = _lcs(x, y)

  def _recon(i, j):
    """private recon calculation"""
    if i == 0 or j == 0:
      return []
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i)]
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    else:
      return _recon(i, j - 1)

  recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
  return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.

  Returns:
    A tuple (f1, precision, recall) for ROUGE-N

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  m = len(reference_words)
  n = len(evaluated_words)
  lcs = _len_lcs(evaluated_words, reference_words)
  return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
  """
  Returns LCS_u(r_i, C) which is the LCS score of the union longest common
  subsequence between reference sentence ri and candidate summary C. For example
  if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
  c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
  "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5". The
  union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5" and
  LCS_u(r_i, C) = 4/5.

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    float: LCS_u(r_i, C)

  ValueError:
    Raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  lcs_union = set()
  reference_words = _split_into_words([reference_sentence])
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    evaluated_words = _split_into_words([eval_s])
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    combined_lcs_length += len(lcs)
    lcs_union = lcs_union.union(lcs)

  union_lcs_count = len(lcs_union)
  union_lcs_value = union_lcs_count / combined_lcs_length
  return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (summary level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # total number of words in reference sentences
  m = len(_split_into_words(reference_sentences))

  # total number of words in evaluated sentences
  n = len(_split_into_words(evaluated_sentences))

  union_lcs_sum_across_all_references = 0
  for ref_s in reference_sentences:
    union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                      ref_s)
  return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(hypotheses, references)
  ]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

  return {
      "rouge_1/f_score": rouge_1_f,
      "rouge_1/r_score": rouge_1_r,
      "rouge_1/p_score": rouge_1_p,
      "rouge_2/f_score": rouge_2_f,
      "rouge_2/r_score": rouge_2_r,
      "rouge_2/p_score": rouge_2_p,
      "rouge_l/f_score": rouge_l_f,
      "rouge_l/r_score": rouge_l_r,
      "rouge_l/p_score": rouge_l_p,
  }


references = [[['<SOS>', 'Which', 'NFL', 'team', 'represented', 'the', 'AFC', 'at', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'Which', 'NFL', 'team', 'represented', 'the', 'NFC', 'at', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'Where', 'did', 'Super', 'Bowl', '50', 'take', 'place', '?', '<EOS>']], [['<SOS>', 'Where', 'did', 'Super', 'Bowl', '50', 'take', 'place', '?', '<EOS>']], [['<SOS>', 'Where', 'did', 'Super', 'Bowl', '50', 'take', 'place', '?', '<EOS>']], [['<SOS>', 'Which', 'NFL', 'team', 'won', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'color', 'was', 'used', 'to', 'emphasize', 'the', '50th', 'anniversary', 'of', 'the', 'Super', 'Bowl', '?', '<EOS>']], [['<SOS>', 'What', 'color', 'was', 'used', 'to', 'emphasize', 'the', '50th', 'anniversary', 'of', 'the', 'Super', 'Bowl', '?', '<EOS>']], [['<SOS>', 'What', 'was', 'the', 'theme', 'of', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'was', 'the', 'theme', 'of', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'was', 'the', 'theme', 'of', 'Super', 'Bowl', '50', '?', '<EOS>']], [['<SOS>', 'What', 'day', 'was', 'the', 'game', 'played', 'on', '?', '<EOS>']]]
candidates = [['defeated', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['Super', '<OOV>', 'title', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['Francisco', '<OOV>', '<OOV>', '?', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'game', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'Levi', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'Stadium', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'the', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'National', '<OOV>', '<OOV>', 'Denver', '<OOV>', 'Broncos', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'National', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>'], ['50th', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', '50th', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['gold', '<OOV>', 'was', '<OOV>', '?', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'prominently', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>'], ['the', '<OOV>', 'the', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'could', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['suspending', '<OOV>', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], ['<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', '50th', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '?', '<OOV>', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'], [',', '2016', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', ',', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>', '<OOV>']]
if __name__ == "__main__":
    assert len(candidates) == len(references)
    # rv = {}
    # for i, v in enumerate(references):
    #     gold_seg = v[0]
    #     pred_seg = candidates[i]
    #     multi_eval(pred_seg, gold_seg, result_dict=rv)
    # for k, v in rv.items():
    #     rv[k] = v / len(candidates)
    # print(rv)
    # print(post_evaluate_test_results_file("con_qa_test_results.txt"))
    # print(post_evaluate_test_results_file("con_crx_test_results.txt"))
