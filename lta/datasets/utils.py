import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union
import random
import numpy as np
import json
from functools import lru_cache


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def action2verbnoun(
        res_action: Tensor,
        class_mappings: dict[tuple[str, str], Tensor]
) -> tuple[Tensor, Tensor]:
    res_verb = torch.matmul(res_action, class_mappings[('verb', 'action')].to(res_action.device))
    res_noun = torch.matmul(res_action, class_mappings[('noun', 'action')].to(res_action.device))
    return res_verb, res_noun


def verbnoun2action(
        res_verb: Tensor,
        res_noun: Tensor,
        verb_noun_to_action: dict[tuple[int, int], int]
) -> Tensor:
    verb_ids, noun_ids = zip(*verb_noun_to_action.keys())

    # Convert to tensors
    verb_ids = torch.tensor(verb_ids, device=res_verb.device)
    noun_ids = torch.tensor(noun_ids, device=res_noun.device)

    # Index into the verb and noun probabilities
    verb_action_probs = res_verb[..., verb_ids]
    noun_action_probs = res_noun[..., noun_ids]

    # Calculate action probabilities
    res_action = verb_action_probs * noun_action_probs  # element-wise multiplication

    return res_action


@lru_cache()
def get_action_matrix(mapping_dict_frozenset, init=-1):
    # Iterate over the frozenset of (key, value) pairs directly
    max_verb_class = max(pair[0][0] for pair in mapping_dict_frozenset) + 1
    max_noun_class = max(pair[0][1] for pair in mapping_dict_frozenset) + 1
    action_matrix = torch.full((max_verb_class, max_noun_class), init,
                               dtype=torch.long)

    for (verb_class, noun_class), action_class in mapping_dict_frozenset:
        action_matrix[verb_class, noun_class] = action_class

    return action_matrix


def to_one_hot(tensor, num_classes):
    # Shape of tensor: [T, 5]
    T, L = tensor.shape

    # Initialize the output tensor with zeros, shape [T, num_classes]
    one_hot_tensor = torch.zeros(T, num_classes, dtype=torch.float32)

    # Create a mask for entries greater than zero
    mask = tensor > 0

    # For rows where all elements are zero (no valid labels), we set the '0' label (background)
    # Since '0' is now a valid label for "no labels", we ensure that frames with all zeros have their '0' index set
    rows_with_all_zeros = ~mask.any(dim=1)
    one_hot_tensor[
        rows_with_all_zeros, 0] = 1  # Set the first class as the "empty" indicator

    # For other cases, filter out the zeros and set the respective indices
    if mask.any():
        labels = tensor[mask]
        indices = torch.repeat_interleave(
            torch.arange(T, device=tensor.device), mask.sum(1))
        one_hot_tensor[indices, labels] = 1

    return one_hot_tensor