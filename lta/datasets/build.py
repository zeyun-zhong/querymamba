import torch
from fvcore.common.registry import Registry

from lta.config import Config


DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets.

The registered object will be called with `obj(**kwargs)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(cfg: Config, mode):
    """
    Build a dataset, defined by `cfg.DATA.DATASET_CLASS`.
    """
    dataset = DATASET_REGISTRY.get(cfg.DATA.DATASET_CLASS)(cfg, mode)
    return dataset


def build_dataloader(cfg: Config, mode):
    assert mode in ['train', 'val', 'test']
    dataset = build_dataset(cfg, mode)

    is_training = mode == "train"

    bs = cfg.TRAIN.BATCH_SIZE if is_training else cfg.VAL.BATCH_SIZE
    workers = cfg.TRAIN.NUM_WORKERS if is_training else cfg.VAL.NUM_WORKERS

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=is_training,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=workers,
        sampler=sampler, pin_memory=True, persistent_workers=True,
    )

    return dataset, dataloader