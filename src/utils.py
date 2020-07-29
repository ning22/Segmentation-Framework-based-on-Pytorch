import random
import numpy as np
import torch
import yaml


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def get_config(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def batch2device(data, device):
    return {k: v if not hasattr(v, 'to') else v.to(device) for k, v in data.items()}