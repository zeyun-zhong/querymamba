import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

import wandb
import os
import yaml
from collections import defaultdict
import pickle
import random

from lta.config import parse_args, load_config, Config
import lta.utils as utils
from lta.datasets import build_dataloader
from lta.models import build_model
from lta.criterion import build_criterion
from helper import *


def launch_job(rank, world_size, cfg: Config):
    utils.set_seed(cfg.SEED)

    utils.setup_logging(level=cfg.LOG_LEVEL)
    logger = utils.get_logger(__name__)
    logger.info(f"| distributed init (world size {world_size})")

    device = rank  # gpu_ids[rank]

    if cfg.TRAIN.ENABLE:
        experiment_name, ckpt_path, save_path = create_ckpt_path(cfg)

        dataset_train, dataloader_train = build_dataloader(cfg, mode="train")
        dataset_val, dataloader_val = build_dataloader(cfg, mode="val")

        model = build_model(
            cfg, num_classes=dataset_train.num_classes, dataset=dataset_train)

        # Print model architecture and trainable params
        utils.print_model(model)
        utils.params_count(model)

        # Load checkpoint
        if cfg.TRAIN.CKPT_PATH is not None:
            ckpt_path = os.path.join(CKPT_PATH, cfg.TRAIN.CKPT_PATH, CKPT_BEST_FNAME)
            load_model(model, ckpt_path)

        model.to(device=device, dtype=get_dtype(cfg.DTYPE))

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], broadcast_buffers=False
        )

        criterion = build_criterion(cfg, dataset_train)

        # Optimizer and learning rate scheduler
        optimizer = build_optimizer(model, cfg)
        lr_scheduler = build_lrscheduler(optimizer, cfg)

        # Metric tracker
        num_action_classes = dataset_train.num_classes["action"]
        metric_tracker = utils.MetricTracker(
            num_classes=(num_action_classes if cfg.MODEL.IGNORE_INDEX < 0
                         else num_action_classes - 1),
            cfg=cfg,
        )

        scaler = None  # GradScaler()
        mixup = None
        if cfg.TRAIN.USE_MIXUP:
            mixup = utils.MixUp(num_classes=dataset_train.num_classes)

        logger.info("Training starts ...")

        best_metric_value = float("inf") if cfg.METRIC_DESCENDING else 0
        for epoch in range(cfg.TRAIN.EPOCHS):
            lr = optimizer.param_groups[-1]['lr']
            for i, param_group in enumerate(optimizer.param_groups):
                logger.info(
                    f"Epoch {epoch + 1} of {cfg.TRAIN.EPOCHS}: "
                    f"param group {i}, learning rate {param_group['lr']}")

            # Initializes all meters
            metric_tracker.reset()
            log_dict = {"lr": lr}
            dataloader_train.sampler.set_epoch(epoch)

            train_one_epoch(
                model, criterion, dataloader_train, optimizer, lr_scheduler,
                metric_tracker, cfg.TRAIN.GRADIENT_CLIPPING, device,
                dtype=get_dtype(cfg.DTYPE), loss_scaler=scaler, mixup=mixup,
                disable_pregress=not utils.is_master_proc(),
            )
            log_dict.update({**metric_tracker.get_all_data(is_training=True)})
            logger.info(metric_tracker.to_string(is_training=True))

            if (epoch + 1) % cfg.VAL.EVALUATE_EVERY == 0:
                evaluate(
                    model, criterion, dataloader_val, metric_tracker, device,
                    dtype=get_dtype(cfg.DTYPE),
                    disable_pregress=not utils.is_master_proc(),
                )
                log_dict.update({**metric_tracker.get_all_data(is_training=False)})
                logger.info(metric_tracker.to_string(is_training=False, idx="all"))

                # Store checkpoint
                metric_cur = metric_tracker.get_data(cfg.PRIMARY_METRIC, False)

                is_best, best_metric_value = save_model(
                    model, optimizer, lr_scheduler,
                    metric_cur, best_metric_value, epoch,
                    cfg.METRIC_DESCENDING,
                    fpath=save_path if cfg.TRAIN.SAVE_MODEL else None,
                )
                logger.info(f"Current metric value: {metric_cur}; best metric value: {best_metric_value}")

            if utils.is_master_proc():
                if epoch == 0:
                    wandb.init(
                        project=(f"UniAnt-{cfg.DATA.DATASET_CLASS}"
                                 if cfg.WANDB_PROJECT is None
                                 else cfg.WANDB_PROJECT),
                        name=experiment_name,
                        mode=None if cfg.USE_WANDB else "disabled",
                    )

                wandb.log(log_dict)
                wandb.summary["val/primary_metric"] = best_metric_value

    if cfg.TEST.ENABLE:
        if cfg.TEST.CKPT_PATH is None:
            logger.warning("No checkpoint path provided.")

        dataset_test, dataloader_test = build_dataloader(cfg, mode="test")

        model = build_model(
            cfg, num_classes=dataset_test.num_classes, dataset=dataset_test)

        # Print model architecture and trainable params
        # utils.print_model(model)
        utils.params_count(model)

        # Load checkpoint
        if cfg.TEST.CKPT_PATH is not None:
            ckpt_path = os.path.join(CKPT_PATH, cfg.TEST.CKPT_PATH, CKPT_BEST_FNAME)
            load_model(model, ckpt_path)

        model.to(device=device, dtype=get_dtype(cfg.DTYPE))

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], broadcast_buffers=False
        )

        criterion = (build_criterion(cfg, dataloader_test)
                     if cfg.DATA.DATASET_CLASS != "Ego4D" else None)

        # Metric tracker
        num_action_classes = dataset_test.num_classes["action"]
        metric_tracker = utils.MetricTracker(
            num_classes=(num_action_classes if cfg.MODEL.IGNORE_INDEX < 0
                         else num_action_classes - 1),
            cfg=cfg,
        ) if cfg.DATA.DATASET_CLASS != "Ego4D" else None

        if metric_tracker is not None:
            metric_tracker.reset()
        log_dict = {}

        evaluate(
            model, criterion, dataloader_test, metric_tracker, device,
            dtype=get_dtype(cfg.DTYPE),
            disable_pregress=not utils.is_master_proc(),
            test_enable=True,
            dataset_name=cfg.DATA.DATASET_CLASS,
        )

        if metric_tracker is not None:
            log_dict.update(
                {**metric_tracker.get_all_data(is_training=False)})
            logger.info(
                metric_tracker.to_string(is_training=False, idx="all"))


def worker(rank, world_size, master_port, args):
    print(f"Worker {rank} starting")
    # distributed setting
    utils.init_distributed_mode(rank, world_size, master_port)
    cfg = load_config(args.cfg_file, args.opts)

    try:
        launch_job(rank, world_size, cfg)
    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt")
    finally:
        print(f"Worker {rank} clean up and shutting down")
        # Ensure clean shutdown of the process group
        destroy_process_group()
        print(f"Worker {rank} has destroyed its process group")


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    master_port = str(random.randint(12000, 31999))

    try:
        mp.spawn(worker, args=(world_size, master_port, args), nprocs=world_size)
    except KeyboardInterrupt:
        print("Main process received KeyboardInterrupt")
    finally:
        print("Main process shutting down.")


if __name__ == "__main__":
    main()