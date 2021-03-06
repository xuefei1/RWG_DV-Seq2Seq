import utils.model_utils as mutil
from utils.lang_utils import *
from data_loaders import DVSeq2SeqDataLoader
from model_dv_seq2seq import *
from data.read_data import *
from constants import *
from params import prepare_params, merge_params
from components import LRDecayOptimizer, get_masked_nll_criterion, get_nll_criterion


def main(params):
    params.model_name = "dv_seq2seq"
    mutil.DEVICE_STR_OVERRIDE = params.device_str
    data_train = read_col_word_delim_data("data/dv_seq2seq_train.txt")
    data_valid = read_col_word_delim_data("data/dv_seq2seq_valid.txt")
    data_test = read_col_word_delim_data("data/dv_seq2seq_test.txt")

    w2c_src = build_w2c_from_seg_word_lists([t[0] + t[1] for t in data_train])
    w2c_tgt = build_w2c_from_seg_word_lists([t[2] for t in data_train], limit=40000)  # limit output vocab size
    print("data_train len: {}".format(len(data_train)))
    print("data_valid len: {}".format(len(data_valid)))
    print("data_test len: {}".format(len(data_test)))
    print("src w2c len: {}".format(len(w2c_src)))
    print("tgt w2c len: {}".format(len(w2c_tgt)))

    pre_built_w2v = None
    src_vocab_cache_file = "cache/dv_seq2seq_src_vocab.pkl"
    tgt_vocab_cache_file = "cache/dv_seq2seq_tgt_vocab.pkl"
    if os.path.isfile(src_vocab_cache_file):
        print("Loading src vocab from cache " + src_vocab_cache_file)
        with open(src_vocab_cache_file, "rb") as f:
            src_vocab = pickle.load(f)
    else:
        print("Building src vocab")
        if pre_built_w2v is None:
            pre_built_w2v = load_gensim_word_vec(params.word_vec_file,
                                                 cache_file=params.vocab_cache_file)
        src_vocab = W2VTrainableVocab(w2c_src, pre_built_w2v, embedding_dim=params.word_embedding_dim, rand_oov_embed=True,
                                      special_tokens=(
                                          PAD_TOKEN,
                                          OOV_TOKEN,
                                      ), light_weight=True)
        with open(src_vocab_cache_file, "wb") as f:
            pickle.dump(src_vocab, f, protocol=4)
    params.src_vocab_size = len(src_vocab.w2i)
    print("src vocab size: ", params.src_vocab_size)

    if os.path.isfile(tgt_vocab_cache_file):
        print("Loading tgt vocab from cache " + tgt_vocab_cache_file)
        with open(tgt_vocab_cache_file, "rb") as f:
            tgt_vocab = pickle.load(f)
    else:
        print("Building tgt vocab")
        if pre_built_w2v is None:
            pre_built_w2v = load_gensim_word_vec(params.word_vec_file,
                                                 cache_file=params.vocab_cache_file)
        tgt_vocab = W2VTrainableVocab(w2c_tgt, pre_built_w2v, embedding_dim=params.word_embedding_dim, rand_oov_embed=False,
                                      special_tokens=(
                                          PAD_TOKEN,
                                          OOV_TOKEN,
                                          SOS_TOKEN,
                                          EOS_TOKEN,
                                      ), light_weight=True)
        with open(tgt_vocab_cache_file, "wb") as f:
            pickle.dump(tgt_vocab, f, protocol=4)
    params.tgt_vocab_size = len(tgt_vocab.w2i)
    print("tgt vocab size: ", params.tgt_vocab_size)

    params.src_w2i = src_vocab.w2i
    params.src_i2w = src_vocab.i2w
    params.tgt_w2i = tgt_vocab.w2i
    params.tgt_i2w = tgt_vocab.i2w
    params.w2i = tgt_vocab.w2i
    params.i2w = tgt_vocab.i2w
    params.pad_idx = tgt_vocab.pad_idx
    params.oov_idx = tgt_vocab.oov_idx
    params.sos_idx = tgt_vocab.w2i[SOS_TOKEN]
    params.eos_idx = tgt_vocab.w2i[EOS_TOKEN]

    print("Preparing data loaders")
    train_loader = DVSeq2SeqDataLoader(params.batch_size, src_vocab, tgt_vocab, src_vocab, data_train)
    valid_loader = DVSeq2SeqDataLoader(params.batch_size, src_vocab, tgt_vocab, src_vocab, data_valid)
    test_loader = DVSeq2SeqDataLoader(params.batch_size, src_vocab, tgt_vocab, src_vocab, data_test)
    print("{} overlapped train/test instances detected".format(len(train_loader.get_overlapping_data(test_loader))))
    print("{} overlapped train/valid instances detected".format(len(train_loader.get_overlapping_data(valid_loader))))
    print("{} overlapped valid/test instances detected".format(len(valid_loader.get_overlapping_data(test_loader))))

    print("Initializing " + params.model_name)
    criterion_gen = get_masked_nll_criterion(len(tgt_vocab))
    criterion_cpy = get_nll_criterion()
    model = make_dv_seq2seq_model(src_vocab.w2v_mat if params.use_pretrained_embedding else None,
                                  tgt_vocab.w2v_mat if params.use_pretrained_embedding else None,
                                  params, len(src_vocab), len(tgt_vocab),
                                  same_word_embedding=params.same_word_embedding)
    model_opt = LRDecayOptimizer(
                    torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lrd_initial_lr,
                                     betas=(params.adam_betas_1, params.adam_betas_2),
                                     eps=params.adam_eps, weight_decay=params.adam_l2),
                    initial_lr=params.lrd_initial_lr, shrink_factor=params.lrd_lr_decay_factor,
                    min_lr=params.lrd_min_lr, past_scores_considered=params.lrd_past_lr_scores_considered,
                    score_method="min", verbose=True, max_fail_limit=params.lrd_max_fail_limit)

    completed_epochs = 0
    best_eval_result = 0
    best_eval_epoch = 0
    past_eval_results = []
    if os.path.isfile(params.saved_model_file):
        print("Found saved model {}, loading".format(params.saved_model_file))
        sd = mutil.model_load(params.saved_model_file)
        saved_params = sd[CHKPT_PARAMS]
        params = merge_params(saved_params, params)
        model.load_state_dict(sd[CHKPT_MODEL])
        model_opt.load_state_dict(sd[CHKPT_OPTIMIZER])
        best_eval_result = sd[CHKPT_BEST_EVAL_RESULT]
        best_eval_epoch = sd[CHKPT_BEST_EVAL_EPOCH]
        past_eval_results = sd[CHKPT_PAST_EVAL_RESULTS]
        completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]

    print(model)
    print("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(n_params))

    if not os.path.isfile(params.saved_model_file) or \
            (os.path.isfile(params.saved_model_file) and params.continue_training):
        print("Training")
        try:
            train_dv_seq2seq(params, model, train_loader, criterion_gen, criterion_cpy, model_opt,
                             completed_epochs=completed_epochs, best_eval_result=best_eval_result,
                             best_eval_epoch=best_eval_epoch, past_eval_results=past_eval_results,
                             eval_loader=valid_loader)
        except KeyboardInterrupt:
            print("training interrupted")

    if len(test_loader) > 0:
        fn = params.saved_models_dir + params.model_name + "_best.pt"
        exclude_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, "", " "]
        if os.path.isfile(fn):
            sd = mutil.model_load(fn)
            completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]
            model.load_state_dict(sd[CHKPT_MODEL])
            print("Loaded best model after {} epochs of training".format(completed_epochs))
        with torch.no_grad():
            model.eval()
            write_line_to_file("input|pred|truth|", f_path=params.model_name + "_test_results.txt")
            for batch in tqdm(test_loader, mininterval=2, desc="Test", leave=False, ascii=True):
                beam_rvs = dv_seq2seq_beam_decode_batch(model, batch, params.sos_idx, tgt_vocab.i2w,
                                                        eos_idx=params.eos_idx,
                                                        len_norm=params.bs_len_norm, gamma=params.bs_div_gamma,
                                                        max_len=params.max_decoded_seq_len,
                                                        beam_width=params.beam_width_test)
                for bi in range(batch[DK_BATCH_SIZE]):
                    msg_str = "".join(batch[DK_SRC_SEG_LISTS][bi])
                    truth_rsp_seg = [w for w in batch[DK_TGT_SEG_LISTS][bi] if w not in exclude_tokens]
                    truth_rsp_str = " ".join(truth_rsp_seg)
                    truth_rsp_str = re.sub(" +", " ", truth_rsp_str)
                    best_rv = [w for w in beam_rvs[bi][0][3] if w not in exclude_tokens]  # word seg list
                    rsp = " ".join(best_rv)
                    write_line_to_file(msg_str + "|" + rsp + "|" + truth_rsp_str,
                                       params.model_name + "_test_results.txt")


if __name__ == "__main__":
    args = prepare_params()
    main(args)
    print("done")
