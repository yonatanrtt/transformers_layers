import shared.constants as constants 
import torch 
import random 
import gc

def init_task(_task):
    db_name = _task 
    if _task == constants.CB:
        from data.datasets.cb_ds import CbDataset 
        from train.tasks.cb_model import CbModel 
        db_class = CbDataset
        model = CbModel 
    elif _task == constants.COPA:
        from data.datasets.copa_ds import CopaDataset 
        from train.tasks.copa_model import CopaModel 
        db_class = CopaDataset
        model = CopaModel 
    else:
        from data.datasets.rte_ds import RteDataset 
        from train.tasks.rte_model import RteModel 
        db_name = constants.RTE 
        db_class = RteDataset
        model = RteModel 
    return db_name, db_class, model

def init_task_baseline(_task):
    db_name = _task 
    if _task == constants.CB:
        from data.datasets.cb_ds import CbDataset 
        from train_baseline.cb_model import CbModelBaseline 
        db_class = CbDataset
        model = CbModelBaseline 
    elif _task == constants.COPA:
        from data.datasets.copa_ds import CopaDataset 
        from train_baseline.copa_model import CopaModelBaseline 
        db_class = CopaDataset
        model = CopaModelBaseline 
    else:
        print("*"*200)
        from data.datasets.rte_ds import RteDataset 
        from train_baseline.rte_model import RteModelBaseline 
        db_name = constants.RTE 
        db_class = RteDataset
        model = RteModelBaseline 
    return db_name, db_class, model

def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def set_all_seeds():
    _seed = constants.SEED 
    gc.collect() 
    torch.cuda.empty_cache()
    random.seed(_seed) 
    torch.manual_seed(_seed) 
    torch.cuda.manual_seed(_seed) 
    torch.cuda.manual_seed_all(_seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def get_sequence_mean(_input, _output):
    attention_mask = _input.attention_mask
    mask_padding = attention_mask.view( [attention_mask.shape[0], attention_mask.shape[1], 1]) * _output
    return mask_padding.sum(axis=1) / attention_mask. sum(axis=1).view([attention_mask. shape[0], 1])

def get_seq_embed(_input, _output, _type=constants.CLS): 
    if _type == constants.CLS:
        return _output[:, 0, :] 
    elif _type == constants.MEAN:
        return get_sequence_mean(_input, _output)

def orthogonal_penalty(_m):
   device = get_device()
   dists = torch.Tensor([torch.dist(_m[i] @ _m[i].T, torch.eye(_m.shape[1]).to(device)) for i in range(_m.shape[1])])
   return torch.mean(dists, axis=0) 

def matrix_dist(_m1, _m2):
    return torch.dist(_m1, _m2)

def matrix_dist_sum(_m1, _m2):
    lst = [] 
    for b in range(_m1.size(0)): 
        for i in range(_m1.size(1)):
            lst.append(_m1[b][i] @ _m2[b][i]) 
    return (sum(lst)) ** 0.5

def matrix_dist_mul(_m1, _m2, _axis):
    mul_layers = _m1 @ _m2
    mul_layers_mean = torch.mean(mul_layers, axis=_axis) 
    b_mul_layers_mean = torch.mean(mul_layers_mean, axis=1) 
    dist_mean = torch.mean(b_mul_layers_mean, axis=0) 
    return dist_mean
