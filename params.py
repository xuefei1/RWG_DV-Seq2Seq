import os
import argparse
import torch

def prepare_params():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    d = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(d + os.path.sep + "cache"): os.makedirs(d + os.path.sep + "cache")
    if not os.path.exists(d + os.path.sep + "saved_models"): os.makedirs(d + os.path.sep + "saved_models")
    if not os.path.exists(d + os.path.sep + "logs"): os.makedirs(d + os.path.sep + "logs")
    parser = argparse.ArgumentParser(description="Input control args for the model")

    # General
    parser.add_argument("-batch_size", help="Batch size", type=int,
                        default=64, required=False)
    parser.add_argument("-epochs", help="Training epochs", type=int,
                        default=70, required=False)
    parser.add_argument("-eval_metric", help="Determining metric name in eval to index performance dict", type=str,
                        default="acc", required=False)
    parser.add_argument("-device_str", help="Device string", type=str,
                        default=None, required=False)
    parser.add_argument("-seed", help="Seed", type=int,
                        default=12345, required=False)
    parser.add_argument("-max_gradient_norm", help="Max grad norm", type=float,
                        default=5.0, required=False)
    parser.add_argument("-past_eval_scores_considered", help="", type=int,
                        default=10, required=False)
    parser.add_argument("-max_decoded_seq_len", help="", type=int,
                        default=20, required=False)
    parser.add_argument("-data_file", help="Data file", type=str,
                        default=d + os.path.sep + "data" + os.path.sep + "cg_word_segs.txt", required=False)
    parser.add_argument("-lazy_load", help="", type=str,
                        default="false", required=False)
    parser.add_argument("-train_data_file", help="", type=str,
                        default=d + os.path.sep + "data" + os.path.sep + "train_pairs.txt", required=False)
    parser.add_argument("-valid_data_file", help="", type=str,
                        default=d + os.path.sep + "data" + os.path.sep + "valid_pairs.txt", required=False)
    parser.add_argument("-test_data_file", help="", type=str,
                        default=d + os.path.sep + "data" + os.path.sep + "test_pairs.txt", required=False)
    parser.add_argument("-w2c_limit", help="", type=int,
                        default=1000000, required=False)
    parser.add_argument("-read_lines_limit", help="", type=int,
                        default=40000000, required=False)
    parser.add_argument("-output_vocab_limit", help="", type=int,
                        default=None, required=False)
    parser.add_argument("-base_k", help="", type=int,
                        default=16, required=False)

    # Adam
    parser.add_argument("-adam_betas_1", help="Beta 1 for Adam optimizer", type=float,
                        default=0.9, required=False)
    parser.add_argument("-adam_betas_2", help="Beta 2 for Adam optimizer", type=float,
                        default=0.999, required=False)
    parser.add_argument("-adam_eps", help="Epsilon for Adam optimizer", type=float,
                        default=1e-8, required=False)
    parser.add_argument("-adam_l2", help="L2 penalty for Adam optimizer", type=float,
                        default=0.0, required=False)

    # NoamOptimizer
    parser.add_argument("-noam_warm_up_steps", help="Warm up steps of the Noam optimizer", type=int,
                        default=4000, required=False)
    parser.add_argument("-noam_factor", help="Factor for Noam optimizer", type=float,
                        default=1.0, required=False)

    # label smoothing criterion
    parser.add_argument("-smoothing_const", help="Label smoothing constant", type=float,
                        default=0.0, required=False)

    # LRDecayOptimizer
    parser.add_argument("-lrd_initial_lr", help="Training initial lr", type=float,
                        default=0.001, required=False)
    parser.add_argument("-lrd_min_lr", help="Training min lr", type=float,
                        default=0.00001, required=False)
    parser.add_argument("-lrd_lr_decay_factor", help="LR decay factor", type=float,
                        default=0.9, required=False)
    parser.add_argument("-lrd_past_lr_scores_considered", help="Past lr loss considered", type=int,
                        default=1, required=False)
    parser.add_argument("-lrd_max_fail_limit", help="Max bad count for lr update", type=int,
                        default=1, required=False)
    parser.add_argument("-lr_decay_with_train_perf", help="", type=str,
                        default="true", required=False)
    parser.add_argument("-lr_decay_with_train_loss_diff", help="", type=str,
                        default="true", required=False)
    parser.add_argument("-train_loss_diff_threshold", help="", type=float,
                        default=0.01, required=False)

    # general embedding
    parser.add_argument("-use_pretrained_embedding", help="Use pre-trained word vectors", type=str,
                        default="true", required=False)
    parser.add_argument("-same_word_embedding", help="Same word embedding", type=str,
                        default="false", required=False)
    parser.add_argument("-filter_embed_further_training", help="Further train word embedding", type=str,
                        default="true", required=False)
    parser.add_argument("-word_embed_further_training", help="Further train word embedding", type=str,
                        default="true", required=False)
    parser.add_argument("-word_embedding_dim", help="Embedding dimension", type=int,
                        default=200, required=False)

    # beam search
    parser.add_argument("-beam_width_test", help="Beam search width when testing", type=int,
                        default=4, required=False)
    parser.add_argument("-beam_width_eval", help="Beam search width when validating", type=int,
                        default=2, required=False)
    parser.add_argument("-beam_width", help="Beam search width", type=int,
                        default=4, required=False)
    parser.add_argument("-bs_len_norm", help="Beam search length norm", type=float,
                        default=0.0, required=False)
    parser.add_argument("-bs_div_gamma", help="Gamma controlling beam search diversity", type=float,
                        default=0.0, required=False)

    # seq2seq
    parser.add_argument("-s2s_teacher_forcing_ratio", help="Seq2seq teacher forcing ratio", type=float,
                        default=1.0, required=False)
    parser.add_argument("-s2s_encoder_hidden_size", help="Encoder hidden size", type=int,
                        default=256, required=False) # if direction is 2, final encoder hidden = this value * 2
    parser.add_argument("-s2s_encoder_dropout_prob", help="Encoder dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-s2s_decoder_dropout_prob", help="Decoder dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-s2s_model_dropout_prob", help="", type=float,
                        default=0.1, required=False)
    parser.add_argument("-s2s_encoder_type", help="Encoder type", type=str,
                        default="gru", required=False)
    parser.add_argument("-s2s_decoder_type", help="Decoder type", type=str,
                        default="gru", required=False)
    parser.add_argument("-s2s_num_encoder_layers", help="Encoder number of layers", type=int,
                        default=1, required=False)
    parser.add_argument("-s2s_num_decoder_layers", help="Decoder number of layers", type=int,
                        default=1, required=False)
    parser.add_argument("-s2s_encoder_rnn_dir", help="RNN direction", type=int,
                        default=2, required=False)

    # rwg
    parser.add_argument("-rwg_dropout_prob", help="Dropout prob", type=float,
                        default=0.0, required=False)
    parser.add_argument("-rwg_hidden_size", help="Hidden size", type=int,
                        default=512, required=False)
    parser.add_argument("-rwg_rnn_dir", help="", type=int,
                        default=2, required=False)
    parser.add_argument("-rwg_rnn_layers", help="", type=int,
                        default=1, required=False)
    parser.add_argument("-rwg_feedforward_hidden_size", help="", type=int,
                        default=1024, required=False)
    parser.add_argument("-rwg_num_attn_heads", help="Number of attention heads", type=int,
                        default=8, required=False)
    parser.add_argument("-rwg_num_encoder_layers", help="Encoder number of layers", type=int,
                        default=2, required=False)
    parser.add_argument("-rwg_rnn_pool_size", help="", type=int,
                        default=1, required=False)

    # dv_seq2seq
    parser.add_argument("-dv_seq2seq_dropout_prob", help="Dropout prob", type=float,
                        default=0.1, required=False)
    parser.add_argument("-dv_seq2seq_hidden_size", help="Hidden size", type=int,
                        default=256, required=False)
    parser.add_argument("-dv_seq2seq_encoder_rnn_dir", help="", type=int,
                        default=2, required=False)
    parser.add_argument("-dv_seq2seq_rnn_layers", help="", type=int,
                        default=1, required=False)

    # bool args
    parser.add_argument("-vis_plot", help="Plot loss with visdom", required=False, action='store_true')
    parser.add_argument("-log_to_file", help="Enables logging important console prints to a file", required=False, action='store_true')
    parser.add_argument("-continue_training", help="Enables continue training", required=False, action='store_true')
    parser.add_argument("-enable_shortlist", help="Reduce data size for faster debugging", required=False, action='store_true')
    parser.add_argument("-lazy_build", help="Lazy building data loaders", required=False, action='store_true')
    parser.add_argument("-no_data_loading", help="", required=False, action='store_true')

    # value args
    parser.add_argument("-dataset_name", help="For selecting datesets", type=str,
                        default="squad", required=False)
    parser.add_argument("-model_name", help="For checkpoint purposes", type=str,
                        default="model", required=False)
    parser.add_argument("-test_play_mode", help="Test play mode", type=str,
                        default="demo", required=False)
    parser.add_argument("-word_delim", help="word delimiter", type=str,
                        default=" ", required=False)
    parser.add_argument("-logs_dir", help="logs directory", type=str,
                        default="logs"+os.path.sep, required=False)
    parser.add_argument("-saved_model_file", help="Checkpoint file name", type=str,
                        default=d+os.path.sep+"saved_models"+os.path.sep+"default_name.pt", required=False)
    parser.add_argument("-saved_criterion_file", help="Checkpoint file name", type=str,
                        default=d + os.path.sep + "saved_models" + os.path.sep + "default_name.pt", required=False)
    parser.add_argument("-saved_optimizer_file", help="Checkpoint file name", type=str,
                        default=d + os.path.sep + "saved_models" + os.path.sep + "default_name.pt", required=False)
    parser.add_argument("-saved_models_dir", help="Checkpoint file folder", type=str,
                        default=d+os.path.sep+"saved_models"+os.path.sep, required=False)
    parser.add_argument("-full_eval_start_epoch", help="Full decoding start at which epoch", type=int,
                        default=3, required=False)
    parser.add_argument("-full_eval_every_epoch", help="Full decoding every how many epochs", type=int,
                        default=1, required=False)
    parser.add_argument("-checkpoint_init_epoch", help="Checkpoint file name start", type=int,
                        default=0, required=False)

    parser.add_argument("-word_vec_file", help="W2V file", type=str,
                        default=d+os.path.sep+"libs"+os.path.sep+"Tencent_AILab_ChineseEmbedding.txt", required=False)
    parser.add_argument("-vocab_cache_file", help="Vocab cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"tencent_ailab_vocab.pkl", required=False)

    parser.add_argument("-train_loader_cache_file", help="Train loader cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"train_loader.pkl", required=False)
    parser.add_argument("-valid_loader_cache_file", help="Valid loader cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"valid_loader.pkl", required=False)
    parser.add_argument("-test_loader_cache_file", help="Test loader cache file", type=str,
                        default=d+os.path.sep+"cache"+os.path.sep+"test_loader.pkl", required=False)

    rv = parser.parse_args()
    rv.use_pretrained_embedding = str2bool(rv.use_pretrained_embedding)
    rv.same_word_embedding = str2bool(rv.same_word_embedding)
    rv.filter_embed_further_training = str2bool(rv.filter_embed_further_training)
    rv.lazy_load = str2bool(rv.lazy_load)
    rv.lr_decay_with_train_perf = str2bool(rv.lr_decay_with_train_perf)
    rv.lr_decay_with_train_loss_diff = str2bool(rv.lr_decay_with_train_loss_diff)

    if torch.cuda.is_available() and rv.seed > 0:
        torch.cuda.manual_seed(rv.seed)
        print('My cuda seed is {0}'.format(torch.cuda.initial_seed()))
    torch.manual_seed(rv.seed)
    print('My seed is {0}'.format(torch.initial_seed()))
    return rv


def merge_params(old_p, new_p):
    for k in old_p.__dict__:
        if old_p.__dict__[k] is not None and k in new_p.__dict__:
            if old_p.__dict__[k] != new_p.__dict__[k]:
                old_p.__dict__[k] = new_p.__dict__[k]
    for k in new_p.__dict__:
        if k not in old_p.__dict__ or old_p.__dict__[k] is None:
            old_p.__dict__[k] = new_p.__dict__[k]
    return old_p
