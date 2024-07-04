"""Save the clip features instead of the video features, since clip features are
much shorter than the whole videos (more efficient to load).
Modified from https://github.com/Evm7/ego4dlta-icvae.
Btw, I think the code of icvae has one issue in line 87 and 129.
To query the features from the whole video at the correct times, we should use clip_parent_start_frame + action_clip_start_frame
instead of just action_clip_start_frame.
"""

import torch, json, os, argparse
import tqdm


def parse_args():
    """
    Parse the following arguments for preprocessing features from Slowfast 8x8 R101
    Args:
        dir_annotations (string): path where the annotations can be found. Default: /data/annotations/
        download_path (string): path where the ego4d features where downloaded into. Default: Ego4d/v1/slowfast8x8_r101_k400/
        features_path (string): path to where the preprocessed features will be stored for model training/testing. Default: Ego4D/features_pad/
        S (int): stride of the feature extractor. Info provided by Ego4d when extracting the downloaded features. Default=16
        N (int): number of features to pad to. Default: 15
        """
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
    parser.add_argument(
        "--download_path",
        help="Path where the ego4d features where downloaded into",
        default="omnivore_video_swinl",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--features_path",
        help="Path to where the preprocessed features will be stored for model training/testing",
        default="omnivore_video_swinl_clips",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--S",
        help="Stride of the feature extractor",
        default=16,
        type=int,
        required=False,
    )
    return parser.parse_args()


def read_json(filename):
    with open(filename) as jsonFile:
        data = json.load(jsonFile)
    return data["clips"]


def read_annotations(dir_annotations):
    """
    Parses the annotations files and maps the videoIDs (name of the downloaded files) with the clipIDs,
     together with the start and end frame assigned for them
    Args:
        dir_annotations: directory where the annotations are found

    Returns: mapped dictionary with all the summary of VidID an ClipID

    """
    split = ["train", "test_unannotated", "val"]  # test, val or train

    clip_list = []
    for s in split:
        entries = read_json(os.path.join(dir_annotations, "fho_lta_{}.json".format(s)))
        clip_list.extend([
            (item["clip_uid"], item["video_uid"], item["clip_parent_start_frame"], item["clip_parent_end_frame"])
            for item in entries
        ])
    clips = set(clip_list)
    return clips


def frame_to_feat(num_frame, S=16):
    """
    Matches the frame start/end from a video to the assigned feature belonging to it
    Args:
        num_frame: frame index
        S: stride assigned by the preprocessor model. Strategy followed when extracting the feature

    Returns: index of feature to filter from/to

    """
    return round(num_frame / S)


def preprocess_features(args):
    root_path = os.path.join(args.root_path, f"v{args.version}")
    clips = read_annotations(os.path.join(root_path, "annotations"))

    features_path = os.path.join(root_path, args.features_path)
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    for clip_uid, video_uid, start_frame, end_frame in tqdm.tqdm(clips):
        feature_start_frame = frame_to_feat(start_frame, args.S)
        feature_end_frame = frame_to_feat(end_frame, args.S)

        # Load features of the whole video
        video_path = os.path.join(root_path, args.download_path, f"{video_uid}.pt")
        video_features = torch.load(video_path)

        clip_features = video_features[feature_start_frame: feature_end_frame]

        store_path = os.path.join(features_path, f"{clip_uid}.pt")
        torch.save(clip_features, store_path)


if __name__ == '__main__':
    args = parse_args()
    preprocess_features(args)
