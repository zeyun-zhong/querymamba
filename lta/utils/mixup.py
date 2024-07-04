# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0.

COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch

from lta.utils.ouput_target_structure import Target
from .logging import get_logger


logger = get_logger(__name__)


def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (tensor): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes. This value is used for
            label smoothing.
    """
    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    """
    This function applies mixup to both one-hot encoded and class index targets.
    Args:
        target (tensor): Target tensor.
        num_classes (int): Total number of classes.
        lam (float): Lambda value for mixup.
        smoothing (float): Label smoothing value.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value

    if target.max() > 1 or target.dim() == 1:  # Class indices
        target = convert_to_one_hot(target, num_classes)
    else:  # already one-hotted
        target[target == 1] = on_value
        target[target == 0] = off_value
    target1 = target * lam
    target2 = target.flip(0) * (1.0 - lam)
    return target1 + target2


class MixUp:
    """
    Apply mixup for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    """

    def __init__(
        self,
        mixup_alpha=0.8,
        mix_prob=1.0,
        label_smoothing=0.1,
        num_classes: dict = {},
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            mix_prob (float): Probability of applying mixup.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (dict): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _get_mixup_params(self):
        lam = 1.0
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                lam = float(lam_mix)
        return lam

    def _mix_batch(self, x, lam):
        if lam == 1.0:
            return

        x_flipped = x.flip(0).mul_(1.0 - lam)
        x.mul_(lam).add_(x_flipped)

    def __call__(self, past, future, target: Target) -> (torch.Tensor, torch.Tensor, Target):
        """
        Apply mixup to input tensor and target.
        Args:
            past (tensor): Input tensor of shape [B, seq_len, D].
            future (tensor): Input tensor of shape [B, seq_len, D].
            past_target (tensor): Target tensor, either one-hot encoded or class indices.
            future_target (tensor): Target tensor, either one-hot encoded or class indices.
        Returns:
            Mixed input tensor and mixed target tensor.
        """
        if len(past) == 1:
            logger.info("Batch size should be greater than 1 for mixup. Current value is 1.")
            return

        lam = self._get_mixup_params()

        self._mix_batch(past, lam)

        target.past_actions = mixup_target(
            target.past_actions, self.num_classes['action'], lam,
            self.label_smoothing
        )

        if future is not None:
            self._mix_batch(future, lam)
            target.future_actions = mixup_target(
                target.future_actions, self.num_classes['action'], lam,
                self.label_smoothing
            )

        if target.past_verbs is not None:
            target.past_verbs = mixup_target(
                target.past_verbs, self.num_classes['verb'], lam, self.label_smoothing
            )
            target.past_nouns = mixup_target(
                target.past_nouns, self.num_classes['noun'], lam, self.label_smoothing
            )

            if target.future_verbs is not None:
                target.future_verbs = mixup_target(
                    target.future_verbs, self.num_classes['verb'], lam, self.label_smoothing
                )
                target.future_nouns = mixup_target(
                    target.future_nouns, self.num_classes['noun'], lam, self.label_smoothing
                )