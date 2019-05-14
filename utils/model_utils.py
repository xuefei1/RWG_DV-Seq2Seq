import copy
import math
import torch
import torch.nn as nn
from constants import *
from utils.misc_utils import write_line_to_file


DEVICE_STR_OVERRIDE = None


def device(ref_tensor=None):
    if ref_tensor is not None:
        return ref_tensor.get_device()
    if DEVICE_STR_OVERRIDE is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(DEVICE_STR_OVERRIDE)


def fix_model_weights(model):
    for param in model.parameters():
        param.requires_grad = False


def unfix_model_weights(model):
    for param in model.parameters():
        param.requires_grad = True


def init_weights(model, base_model_type="rnn"):
    # For now, only handle rnn differently
    if base_model_type == "rnn":
        for p in filter(lambda pa: pa.requires_grad, model.parameters()):
            if p.dim() == 1:
                p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
            else:
                nn.init.xavier_normal_(p, math.sqrt(3))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def parallel(model):
    if hasattr(model, "avoid_parallel") and model.avoid_parallel:
        return model
    return nn.DataParallel(model)


def _save_model(file_path, epoch, model, optimizer, metadata={}):
    sv = {
        CHKPT_COMPLETED_EPOCHS: epoch,
        CHKPT_MODEL: model.state_dict(),
        CHKPT_OPTIMIZER: optimizer.state_dict(),
        CHKPT_METADATA: metadata
    }
    torch.save(sv, file_path)


def load_transformer_dict(file_path):
    return torch.load(file_path, map_location=lambda storage,loc:storage)


def model_checkpoint(file_path, epoch, model, optimizer, params,
                     past_eval_results, best_eval_result, best_eval_epoch, metadata={}):
    print("saving model to {}, please do not terminate".format(file_path))
    sv = {
        CHKPT_COMPLETED_EPOCHS: epoch,
        CHKPT_MODEL: model.state_dict(),
        CHKPT_OPTIMIZER: optimizer.state_dict(),
        CHKPT_PARAMS: params,
        CHKPT_PAST_EVAL_RESULTS: past_eval_results,
        CHKPT_BEST_EVAL_RESULT: best_eval_result,
        CHKPT_BEST_EVAL_EPOCH: best_eval_epoch,
        CHKPT_METADATA: metadata
    }
    torch.save(sv, file_path)
    print("checkpoint done")


def model_load(file_path):
    return torch.load(file_path, map_location=lambda storage,loc:storage)


def check_nan(tsr, chkpt=0, terminate=True):
    if not isinstance(tsr, torch.Tensor): return False
    if torch.isnan(tsr).any():
        if terminate:
            assert False, "nan chkpt {}: nan detected in tensor".format(chkpt)
        else:
            print("nan chkpt {}: nan detected in tensor".format(chkpt))
        return True
    return False


def inspect_tensor(tsr, name="tensor", logs_dir="logs/", verbose=False):
    opf = logs_dir + name + "_report.txt"
    torch.set_printoptions(threshold=5000)
    _output_tensor_stats(tsr, opf, verbose=verbose)
    check_nan(tsr, chkpt=name)


def _output_tensor_stats(tsr, opf, verbose=False):
    write_line_to_file(str(tsr), opf, verbose=verbose)
    max_idx, max_val = tsr.topk(1)
    write_line_to_file("max idx in tsr: {}".format(max_idx.item()), opf, verbose=verbose)
    write_line_to_file("max val in tsr: {}".format(max_val.item()), opf, verbose=verbose)
    min_val = tsr.min(1)
    write_line_to_file("min val in tsr: {}".format(min_val.item()), opf, verbose=verbose)
    unique_vals = torch.unique(tsr, sorted=True).tolist()
    write_line_to_file("unique values in tensor: {}".format(str(unique_vals)), verbose=verbose)
    n_zeros = torch.sum((tsr == 0).int()).item()
    write_line_to_file("number of zeros in tensor: {}".format(n_zeros), verbose=verbose)


def inspect_tensors(named_tuples, fn="tensors", logs_dir="logs/", verbose=False):
    opf = logs_dir + fn + "_report.txt"
    torch.set_printoptions(threshold=5000)
    for name, tsr in named_tuples:
        write_line_to_file(name, opf, verbose=verbose)
        _output_tensor_stats(tsr, opf, verbose=verbose)
        write_line_to_file("", opf, verbose=verbose)
        check_nan(tsr, chkpt=name)


def count_zero_weights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data[0]
    return zeros


def inspect_model(model, name="model", logs_dir="logs/", verbose=False):
    opf = logs_dir + name + "_report.txt"
    write_line_to_file("model name: {}".format(name), opf, verbose=verbose)
    _output_component_stats(model, opf, verbose=verbose)


def _output_component_stats(net, opf, verbose=False):
    write_line_to_file(str(net), opf, verbose=verbose)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    write_line_to_file("number of trainable parameters: {}".format(n_params), opf, verbose=verbose)
    zero_weights = count_zero_weights(net)
    write_line_to_file("current zeros weights: {}".format(zero_weights), opf, verbose=verbose)


def step_with_check(step_func, model, terminate=True, report_dict={}):
    a = copy.deepcopy(list(filter(lambda pa: pa[1].requires_grad, model.named_parameters())))
    step_func()
    b = copy.deepcopy(list(filter(lambda pa: pa[1].requires_grad, model.named_parameters())))
    not_changed = []
    named_comps = []
    for i, tup in enumerate(a):
        name1, p1 = tup
        name2, p2 = b[i]
        assert name1 == name2
        same = torch.equal(p1.data, p2.data)
        not_changed.append(same)
        named_comps.append([name1, name2, same])
        if same:
            if name1 not in report_dict: report_dict[name1] = 0
            report_dict[name1] += 1
    if all(not_changed):
        if terminate: assert False, "error: all weights not updated after step()"
        else: print("warning: all weights not updated after step()")
    return named_comps
