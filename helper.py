import torch
from torch import Tensor, nn
from tqdm import tqdm
import time
import os
from torch.cuda.amp import autocast
from contextlib import nullcontext
from typing import Optional, Tuple

import lta.utils as utils
from lta.utils import Target, Prediction
from lta.config import Config
from lta.criterion import Criterion_Ego4D
from lta.datasets.utils import dump_json
from ego4d_test_set import generate_outputs

__all__ = [
    "get_dtype",
    "train_one_epoch",
    "evaluate",
    "build_optimizer",
    "build_lrscheduler",
    "create_ckpt_path",
    "save_model",
    "load_model",
    "CKPT_PATH",
    "CKPT_BEST_FNAME",
]

CKPT_PATH = 'checkpoints'
CKPT_BEST_FNAME = 'checkpoint_best.pth'
logger = utils.get_logger(__name__)


def get_dtype(str_type):
    dtype_dict = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_dict[str_type]


def get_batch_size(data: dict):
    for key, val in data.items():
        if isinstance(val, Tensor):
            return val.size(0)
        if isinstance(val, list):
            return len(val)


def train_one_epoch(
        model,
        criterion: Criterion_Ego4D,
        data_loader,
        optimizer,
        scheduler,
        metric_tracker,
        grad_clip,
        device,
        dtype=torch.float32,
        loss_scaler=None,
        mixup: utils.MixUp = None,
        disable_pregress=False,
):
    model.train()
    # Conditionally use autocast or the dummy context manager
    context = autocast(dtype=dtype) if dtype != torch.float32 else nullcontext()

    for data in tqdm(data_loader, desc="Training", disable=disable_pregress):
        data = {
            key: val.to(device=device)
            if val is not None and not isinstance(val, list)
            else val
            for key, val in data.items()
        }

        batch_size = get_batch_size(data)
        past_feats = data.get("past_feats", None)
        future_feats = data.get("future_feats", None)
        future_verb = data.get("future_verb", None)
        future_noun = data.get("future_noun", None)
        past_verbs = data.get("past_verb", None)
        past_nouns = data.get("past_noun", None)

        target = Target(
            past_feats=past_feats,
            future_feats=future_feats,
            past_actions=data.get("past_act", None),
            past_verbs=past_verbs,
            past_nouns=past_nouns,
            future_actions=data.get("future_act", None),
            future_verbs=future_verb,
            future_nouns=future_noun,
            vid_name=data.get("vid_name", None),
            work_indices=data.get("work_indices", None),
            num_frames=data.get("num_frames", None),
        )

        # Mixup in place
        if mixup is not None:
            mixup(past_feats, future_feats, target)
            target.mixup_enabled = True

        with context:
            pred: Prediction = model(past_feats)
            loss, loss_dict = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()

        # Clip the gradients if required
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        metric_tracker.update(loss_dict, batch_size, is_training=True)

    # gather the stats from all processes
    metric_tracker.synchronize_between_processes(is_training=True)

    if scheduler is not None:
        scheduler.step()


@torch.inference_mode()
def evaluate(
        model,
        criterion: Optional[Criterion_Ego4D],
        data_loader,
        metric_tracker: Optional,
        device,
        dtype=torch.float32,
        disable_pregress=False,
        test_enable=False,
        dataset_name="Ego4D",
        max_len: Optional[int] = None,
):
    model.eval()
    # Conditionally use autocast or the dummy context manager
    context = autocast(dtype=dtype) if dtype != torch.float32 else nullcontext()
    out = {}

    for data in tqdm(data_loader, desc="Evaluation", disable=disable_pregress):
        data = {
            key: val.to(device=device)
            if val is not None and not isinstance(val, list)
            else val
            for key, val in data.items()
        }

        batch_size = get_batch_size(data)
        past_feats = data.get("past_feats", None)
        future_feats = data.get("future_feats", None)
        future_verb = data.get("future_verb", None)
        past_verbs = data.get("past_verb", None)
        past_nouns = data.get("past_noun", None)

        target = Target(
            past_feats=past_feats,
            future_feats=future_feats,
            past_actions=data.get("past_act", None),
            past_verbs=past_verbs,
            past_nouns=past_nouns,
            future_actions=data.get("future_act", None),
            future_verbs=data.get("future_verb", None),
            future_nouns=data.get("future_noun", None),
            vid_name=data.get("vid_name", None),
            work_indices=data.get("work_indices", None),
            num_frames=data.get("num_frames", None),
        )

        with context:
            pred: Prediction = model(past_feats, future_feats)

            if criterion is not None:
                loss, loss_dict = criterion(pred, target, is_training=False)

        # We generate outputs for Ego4d
        if test_enable and dataset_name in ["Ego4D"]:
            out_cur = generate_outputs(
                pred.future_verbs,
                pred.future_nouns,
                data_loader.dataset.verb_noun_cooccurrences,
                data["clip_uid_action_idx"],
            )
            out.update(out_cur)

        if criterion is not None:
            metric_tracker.update(loss_dict, batch_size, is_training=False)

    if out:
        os.makedirs("output", exist_ok=True)
        dump_json(out, "output/output.json")

    if criterion is not None:
        # gather the stats from all processes
        metric_tracker.synchronize_between_processes(is_training=False)


def create_ckpt_path(cfg: Config):
    time_cur = time.strftime('%Y%m%d-%H:%M:%S')

    experiment_name = f'{cfg.MODEL.MODEL_CLASS}-{cfg.MODEL.N_LAYER}-{cfg.MODEL.D_MODEL}_' \
                      f'bs{cfg.TRAIN.BATCH_SIZE}_lr{cfg.TRAIN.LR}_' \
                      f'wd{cfg.TRAIN.WEIGHT_DECAY}_' \
                      f'{time_cur}'
    if cfg.NOTE is not None:
        experiment_name = f'{cfg.NOTE}_{experiment_name}'

    ckpt_path = os.path.join(CKPT_PATH, experiment_name)
    ckpt_save_path = os.path.join(ckpt_path, CKPT_BEST_FNAME)
    return experiment_name, ckpt_path, ckpt_save_path


def build_optimizer(model, cfg: Config):
    optimizer_name = cfg.TRAIN.OPTIMIZER
    params = model.parameters()

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            nesterov=True, momentum=0.9)
    else:
        raise NotImplementedError(f'Optimizer {optimizer_name} not supported!')
    return optimizer


def build_lrscheduler(optimizer, cfg: Config):
    scheduler_name = cfg.TRAIN.SCHEDULER
    if scheduler_name is None or scheduler_name in ["NONE", "none", "None"]:
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = utils.WarmUpCosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.EPOCHS,
            warmup_epochs=cfg.TRAIN.WARMUP_STEPS,
            eta_min=cfg.TRAIN.MIN_LR,
        )
    else:
        raise NotImplementedError(
            f'LRScheduler {scheduler_name} not supported!')
    return scheduler


def save_model(
        model, optimizer, scheduler,
        metric_cur: float, metric_best: float, epoch: int,
        metric_descending=False, fpath: Optional[str] = None,
        save_each_after_epoch=0,
) -> Tuple[bool, float]:
    save = False

    if metric_descending:
        if metric_cur < metric_best:
            metric_best = metric_cur
            save = True
    else:
        if metric_cur > metric_best:
            metric_best = metric_cur
            save = True

    if fpath is None:
        return save, metric_best

    if save:
        store_checkpoint(model, optimizer, scheduler, epoch, fpath)

    if epoch > save_each_after_epoch:
        path = fpath.replace(CKPT_BEST_FNAME, f"checkpoint_epoch_{epoch:03}")
        store_checkpoint(model, optimizer, scheduler, epoch, path)

    return save, metric_best


def store_checkpoint(model, optimizer, scheduler,
                     epoch: int, fpath: Optional[str] = None) -> None:
    model_without_ddp = model
    if isinstance(model, nn.parallel.DistributedDataParallel) or isinstance(
            model, nn.parallel.DataParallel):
        model_without_ddp = model.module
    checkpoint = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler':
            scheduler.state_dict()
            if scheduler is not None
            else None,
        'epoch': epoch,
    }

    ckpt_path = "/".join(fpath.split("/")[:-1])
    os.makedirs(ckpt_path, exist_ok=True)

    logger.info(f'Storing ckpt at epoch {epoch} to {fpath}')
    if utils.is_master_proc():
        torch.save(checkpoint, fpath)


def load_model(model, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pretrained_dict = checkpoint['model']
    missing_keys, unexp_keys = model.load_state_dict(
        pretrained_dict, strict=False)

    if not missing_keys and not unexp_keys:
        logger.info(f'Loaded model from {ckpt_path}')
        return

    logger.warning('Could not init from %s: %s', ckpt_path, missing_keys)
    logger.warning('Unused keys in %s: %s', ckpt_path, unexp_keys)
