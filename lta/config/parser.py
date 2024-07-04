"""Argument parser functions."""

import argparse
import yaml
from typing import List, Optional

from .default_config import Config


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    """
    parser = argparse.ArgumentParser(
        description="UniAnt training and testing pipeline."
    )
    parser.add_argument(
        "--port",
        default="12355",
        help="Master port for DDP",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/ek100/UniAnt.yaml",
    )
    parser.add_argument(
        "--opts",
        help="See config/default_config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(path_to_config: Optional[str] = None, opts: Optional[List[str]] = None) -> Config:
    cfg = Config()
    if path_to_config:
        with open(path_to_config, 'r') as file:
            data = yaml.safe_load(file)
        cfg.merge_updates(data)
    if opts:
        cfg.merge_opts(opts)
    return cfg