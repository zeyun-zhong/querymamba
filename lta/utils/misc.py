import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import yaml
from fvcore.common.config import CfgNode

import lta.utils.logging as logging


logger = logging.get_logger(__name__)


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f'Set SEED to {SEED}')


def exists(val):
    return val is not None


def tonumpy(*tensor):
    """Convert a list of torch Tensors to a list of numpy arrays."""
    return [
        ts.detach().cpu().numpy()
        if isinstance(ts, torch.Tensor)
        else ts for ts in tensor
    ]


def cfg2dict(cfg):
    """Convert a cfg node to a dictionary recursively"""
    out = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            out[key] = cfg2dict(val)
        else:
            out[key] = val
    return out


def load_default_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        if isinstance(v, (list, tuple)):
            parser.add_argument(f"--{k}", nargs='+', type=float)
            continue

        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(
        '{:f}'.format(num).rstrip('0').rstrip('.'),
        ['', 'K', 'M', 'B', 'T'][magnitude]
    )


def params_count(*models):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    for model in models:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{model.__class__.__name__} training weights: {human_format(total_params)}")


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def print_model(model):
    parnum_all = 0
    for n, p in model.named_parameters():
        parnum = sum(pa.numel() for pa in p if pa.requires_grad)
        parnum_all = parnum_all + parnum if p.requires_grad else parnum_all
        logger.info(f"{n:75} {human_format(parnum):8}")
    logger.info(f"All training weights: {human_format(parnum_all)}")


def cfg_to_dict(cfg_node):
    """Convert CfgNode instance to dictionary."""
    config_dict = {}
    for key in cfg_node:
        if isinstance(cfg_node[key], CfgNode):
            config_dict[key] = cfg_to_dict(cfg_node[key])
        else:
            config_dict[key] = cfg_node[key]
    return config_dict