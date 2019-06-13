import pickle
import utils.model_utils as mutil
from utils.lang_utils import *
from utils.model_utils import model_load
from data_loaders import RWGDataLoader
from model_rwg import *
from data.read_data import *
from constants import *
from params import prepare_params, merge_params


def main(params):
    params.model_name = "rwg"
    mutil.DEVICE_STR_OVERRIDE = params.device_str
    data_train = read_col_word_delim_data("data/rwg_train.txt")
    data_valid = read_col_word_delim_data("data/rwg_valid.txt")
    data_test = read_col_word_delim_data("data/rwg_test.txt")

    w2c_src = build_w2c_from_seg_word_lists([t[0] for t in data_train])
    w2c_tgt = build_w2c_from_seg_word_lists([t[1] for t in data_train])
    print("data_train len: {}".format(len(data_train)))
    print("data_valid len: {}".format(len(data_valid)))
    print("data_test len: {}".format(len(data_test)))
    print("src w2c len: {}".format(len(w2c_src)))
    print("tgt w2c len: {}".format(len(w2c_tgt)))

    pre_built_w2v = None
    src_vocab_cache_file = "cache/rwg_src_vocab.pkl"
    tgt_vocab_cache_file = "cache/rwg_tgt_vocab.pkl"
    if os.path.isfile(src_vocab_cache_file):
        print("Loading src vocab from cache " + src_vocab_cache_file)
        src_vocab = pickle.load(open(src_vocab_cache_file, "rb"))
    else:
        print("Building src vocab")
        if pre_built_w2v is None:
            pre_built_w2v = load_gensim_word_vec(params.word_vec_file,
                                                 cache_file=params.vocab_cache_file)
        src_vocab = W2VTrainableVocab(w2c_src, pre_built_w2v, embedding_dim=params.word_embedding_dim,
                                      special_tokens=(
                                          PAD_TOKEN,
                                          OOV_TOKEN,
                                      ), light_weight=True)
        pickle.dump(src_vocab, open(src_vocab_cache_file, "wb"), protocol=4)
    params.src_vocab_size = len(src_vocab.w2i)
    print("src vocab size: ", params.src_vocab_size)

    if os.path.isfile(tgt_vocab_cache_file):
        print("Loading tgt vocab from cache " + tgt_vocab_cache_file)
        tgt_vocab = pickle.load(open(tgt_vocab_cache_file, "rb"))
    else:
        print("Building tgt vocab")
        if pre_built_w2v is None:
            pre_built_w2v = load_gensim_word_vec(params.word_vec_file,
                                                 cache_file=params.vocab_cache_file)
        tgt_vocab = W2VTrainableVocab(w2c_tgt, pre_built_w2v, embedding_dim=params.word_embedding_dim,
                                      special_tokens=(
                                          PAD_TOKEN,
                                          OOV_TOKEN,
                                          SOS_TOKEN,
                                          EOS_TOKEN,
                                      ), light_weight=True)
        pickle.dump(tgt_vocab, open(tgt_vocab_cache_file, "wb"), protocol=4)
    params.tgt_vocab_size = len(tgt_vocab.w2i)
    print("tgt vocab size: ", params.tgt_vocab_size)

    if params.same_word_embedding:
        params.tgt_vocab_size = params.src_vocab_size

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
    params.src_oov2wi = src_vocab.get_oov2wi_dict()

    print("Preparing data loaders")
    train_loader = RWGDataLoader(params.batch_size, src_vocab, tgt_vocab, data_train)
    valid_loader = RWGDataLoader(params.batch_size, src_vocab, tgt_vocab, data_valid)
    test_loader = RWGDataLoader(params.batch_size, src_vocab, tgt_vocab, data_test)
    print("{} overlapped train/test instances detected".format(len(train_loader.get_overlapping_data(test_loader))))
    print("{} overlapped train/valid instances detected".format(len(train_loader.get_overlapping_data(valid_loader))))
    print("{} overlapped valid/test instances detected".format(len(valid_loader.get_overlapping_data(test_loader))))

    print("Initializing " + params.model_name)
    criterion = get_masked_bce_criterion(params.tgt_vocab_size, pad_idx=tgt_vocab.pad_idx)
    model = make_rwg_model(src_vocab.w2v_mat if params.use_pretrained_embedding else None,
                           params, params.src_vocab_size, params.tgt_vocab_size,
                           len(src_vocab.oov_w2i), use_log_softmax=False)
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
        sd = model_load(params.saved_model_file)
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
            train_rwg(params, model, train_loader, criterion, model_opt,
                      completed_epochs=completed_epochs, best_eval_result=best_eval_result,
                      best_eval_epoch=best_eval_epoch, past_eval_results=past_eval_results,
                      eval_loader=valid_loader)
        except KeyboardInterrupt:
            print("training interrupted")

    if len(test_loader) > 0:
        # load best model if possible
        fn = params.saved_models_dir + params.model_name + "_best.pt"
        if os.path.isfile(fn):
            sd = model_load(fn)
            completed_epochs = sd[CHKPT_COMPLETED_EPOCHS]
            model.load_state_dict(sd[CHKPT_MODEL])
            print("Loaded best model after {} epochs of training".format(completed_epochs))
        with torch.no_grad():
            model = model.to(device())
            criterion = criterion.to(device())
            model.eval()
            run_rwg_epoch(test_loader, model, criterion, None,
                          pad_idx=params.pad_idx, model_name=params.model_name,
                          report_acc=True, curr_epoch=1,
                          logs_dir=None, desc="Test")


if __name__ == "__main__":
    args = prepare_params()
    main(args)
    print("done")
