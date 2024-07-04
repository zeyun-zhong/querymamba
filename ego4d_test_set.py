import torch
from einops import rearrange
from collections import defaultdict


def compute_joint_distribution(verb_probs, noun_probs, co_occurrence_matrix):
    # Assume verb_probs is [batch_size, seq_len, num_verbs]
    # Assume noun_probs is [batch_size, seq_len, num_nouns]
    # co_occurrence_matrix is [num_verbs, num_nouns]

    # Expand verb and noun probabilities for broadcasting
    verb_probs_expanded = verb_probs.unsqueeze(-1)  # Shape: [batch_size, seq_len, num_verbs, 1]
    noun_probs_expanded = noun_probs.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, num_nouns]

    # Calculate joint probabilities using broadcasting
    joint_probs = verb_probs_expanded * noun_probs_expanded * co_occurrence_matrix

    # Normalize joint probabilities across all verb-noun pairs
    joint_probs = joint_probs / joint_probs.sum(dim=(-2, -1), keepdim=True)

    return joint_probs


def sample_verb_noun_pairs(joint_probs):
    # joint_probs is [batch_size, seq_len, num_verbs, num_nouns]
    batch_size, seq_len, num_verbs, num_nouns = joint_probs.shape

    # Flatten the last two dimensions for sampling
    joint_probs_flat = rearrange(joint_probs, "b t v n -> (b t) (v n)")

    # Sample indices
    indices = torch.multinomial(joint_probs_flat, num_samples=5,
                                replacement=True)

    indices = rearrange(indices, "(b t) n -> b t n", b=batch_size)

    # Convert flat indices to verb and noun indices
    verb_indices = indices // num_nouns
    noun_indices = indices % num_nouns

    return verb_indices, noun_indices


def generate_outputs(
        verb_logits,
        noun_logits,
        co_occurrence_matrix,
        clip_uid_action_idx: list[str]
):
    co_occurrence_matrix = co_occurrence_matrix.to(verb_logits.device)

    # Let's remove the background class
    verb_logits = verb_logits[..., 1:]
    noun_logits = noun_logits[..., 1:]
    co_occurrence_matrix = co_occurrence_matrix[1:, 1:]

    verb_probs = verb_logits.softmax(dim=-1)
    noun_probs = noun_logits.softmax(dim=-1)
    joint_probs = compute_joint_distribution(verb_probs, noun_probs, co_occurrence_matrix)
    verb_indices, noun_indices = sample_verb_noun_pairs(joint_probs)

    out = defaultdict(dict)

    for i, key in enumerate(clip_uid_action_idx):
        verbs = [verb_indices[i, :, j].tolist() for j in range(5)]
        nouns = [noun_indices[i, :, j].tolist() for j in range(5)]
        out[key]["verb"] = verbs
        out[key]["noun"] = nouns

    return out


if __name__ == '__main__':
    B, T, n_verb, n_noun = 1, 3, 5, 6
    verb_probs = torch.randn((B, T, n_verb)).softmax(-1)
    noun_probs = torch.randn((B, T, n_noun)).softmax(-1)
    co_occurrence_matrix = torch.randint(0, 100, (n_verb, n_noun))
    clip_uid_action_idx = ["action_1"]
    verb_indices, noun_indices = generate_outputs(
        verb_probs, noun_probs, co_occurrence_matrix, clip_uid_action_idx)

