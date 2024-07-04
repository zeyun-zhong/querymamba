import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Optional
import numpy as np
import editdistance


@torch.inference_mode()
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions
    for the specified values of k
    Args:
        output (*, K) predictions
        target (*, ) targets
    """
    # flatten the initial dimensions, to deal with 3D+ input
    output = output.flatten(0, -2)

    if target.ndim == 3:
        target = target.argmax(dim=-1)

    target = target.flatten()
    # Now compute the accuracy
    maxk = max(topk)
    counts = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k * (100.0 / counts))
    return res, counts


def edit_distance(
        preds,  # (B, T, C)
        labels
):
    """
    We compute Edit@20
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    if preds.dim() == 3:
        preds = preds.argmax(-1)

    preds = preds.unsqueeze(-1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)


