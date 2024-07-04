import os
import torch
import argparse
import json


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide details of the preprocessment of the Ego4d downloaded features."
    )
    parser.add_argument(
        "--root_path",
        default="/home/zhong/Documents/datasets/ego4d",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--version",
        default=1,
        type=int,
        required=False,
    )
    return parser.parse_args()


def frame_to_feat(num_frame, stride=16):
    """
    Matches the frame start/end from a video to the assigned feature belonging to it
    Args:
        num_frame: frame index
        S: stride assigned by the preprocessor model. Strategy followed when extracting the feature

    Returns: index of feature to filter from/to

    """
    return round(num_frame / stride)


def list2tensor(lists, max_len, max_cls=15):
    tensor = torch.zeros(max_len, max_cls, dtype=torch.int) - 1
    # Fill the tensor with values from the lists
    for i, sublist in enumerate(lists):
        if len(sublist) > 0:
            tensor[i, :len(sublist)] = torch.tensor(list(sublist), dtype=torch.int)
    return tensor


def generate_ego4d_actions(root_path="/home/zhong/Documents/datasets/ego4d/", version=1):
    train_path = [f"v{version}/annotations/fho_lta_train.json"]
    val_path = [f"v{version}/annotations/fho_lta_val.json"]

    train_clips = []
    for i in range(len(train_path)):
        train_anno = load_json(os.path.join(root_path, train_path[i]))
        train_clips.extend(train_anno["clips"])

    val_clips = []
    for i in range(len(val_path)):
        val_anno = load_json(os.path.join(root_path, val_path[i]))
        val_clips.extend(val_anno["clips"])

    train_clips.extend(val_clips)

    action_taxonomy = {}
    for clip in train_clips:
        verb, noun = clip["verb_label"], clip["noun_label"]

        if f"{verb},{noun}" in action_taxonomy:
            action_taxonomy[f"{verb},{noun}"]["freq"] += 1
            continue

        action_taxonomy[f"{verb},{noun}"] = {
            "verb": clip["verb"],
            "noun": clip["noun"],
            "freq": 1,
        }

    # Sorting the keys based on verb_label first and then noun_label
    sorted_keys = sorted(action_taxonomy.keys())

    # If you need to create a new dictionary with sorted keys
    sort_taxonomy = {key: action_taxonomy[key] for key in sorted_keys}

    action_taxonomy = {}
    for i, (key, val) in enumerate(sort_taxonomy.items()):
        val.update({"action_label": i})
        action_taxonomy[key] = val

    save_path = os.path.join(root_path, f"v{version}/annotations/action_taxonomy.json")
    dump_json(action_taxonomy, save_path)

    print(f"Saved action taxonomy as {save_path}.")


def generate_ego4d_gts(root_path="/home/zhong/Documents/datasets/ego4d/", version=1):
    root_path = os.path.join(root_path, f"v{version}")
    train_path = os.path.join(root_path, "annotations/fho_lta_train.json")
    val_path = os.path.join(root_path, "annotations/fho_lta_val.json")
    test_path = os.path.join(root_path, "annotations/fho_lta_test_unannotated.json")

    action_taxonomy = load_json(os.path.join(root_path, "annotations/action_taxonomy.json"))

    train_anno = load_json(train_path)
    val_anno = load_json(val_path)

    train_val_clips = train_anno["clips"]
    train_val_clips.extend(val_anno["clips"])

    clips_all = []
    clips = []
    clip_uid_cur = "hhh"
    for clip in train_val_clips:
        if clip["clip_uid"] != clip_uid_cur:
            clip_uid_cur = clip["clip_uid"]
            if len(clips) > 0:
                clips_all.append(clips)
            clips = []
        clips.append(clip)

    for clips in clips_all:
        clip_uid = clips[0]["clip_uid"]
        max_frame = max([clip["action_clip_end_frame"] for clip in clips])
        max_feat_frame = frame_to_feat(max_frame)

        # init annos
        verbs = [set() for _ in range(max_feat_frame)]
        nouns = [set() for _ in range(max_feat_frame)]
        actions = [set() for _ in range(max_feat_frame)]

        for clip in clips:
            start_feat_frame = frame_to_feat(clip["action_clip_start_frame"])
            end_feat_frame = frame_to_feat(clip["action_clip_end_frame"])

            verb_label = clip["verb_label"]
            noun_label = clip["noun_label"]
            action_label = action_taxonomy[f"{verb_label},{noun_label}"]["action_label"]

            for i in range(start_feat_frame, end_feat_frame):
                verbs[i].add(verb_label)
                nouns[i].add(noun_label)
                actions[i].add(action_label)

        try:
            verbs = list2tensor(verbs, max_feat_frame)
            nouns = list2tensor(nouns, max_feat_frame)
            actions = list2tensor(actions, max_feat_frame)
        except Exception as e:
            print(e)
            exit()

        verb_path = os.path.join(root_path, "verb_anno_perframe")
        noun_path = os.path.join(root_path, "noun_anno_perframe")
        action_path = os.path.join(root_path, "action_anno_perframe")
        os.makedirs(verb_path, exist_ok=True)
        os.makedirs(noun_path, exist_ok=True)
        os.makedirs(action_path, exist_ok=True)

        torch.save(verbs, os.path.join(verb_path, f"{clip_uid}.pt"))
        torch.save(nouns, os.path.join(noun_path, f"{clip_uid}.pt"))
        torch.save(actions, os.path.join(action_path, f"{clip_uid}.pt"))


if __name__ == '__main__':
    args = parse_args()
    root_path, version = args.root_path, args.version
    generate_ego4d_actions(root_path, version)
    generate_ego4d_gts(root_path, version)