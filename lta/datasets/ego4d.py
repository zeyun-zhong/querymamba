"""Ego4D feature dataset"""
import os
from multiprocessing import Manager
import collections
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

import lta.datasets.utils as utils
from lta.utils import get_logger
from lta.datasets.build import DATASET_REGISTRY
from lta.config import Config


logger = get_logger(__name__)


def frame_to_feat(num_frame, stride=16):
    """
    Matches the frame start/end from a video to the assigned feature belonging to it
    Args:
        num_frame: frame index
        S: stride assigned by the preprocessor model. Strategy followed when extracting the feature

    Returns: index of feature to filter from/to

    """
    return round(num_frame / stride)


@DATASET_REGISTRY.register()
class Ego4D(Dataset):

    def __init__(self, cfg: Config, mode: str):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.n_obs = 8  # default for ego4d, but does not matter here
        self.n_future = 20
        self.feature_stride = 16

        self.root_path = os.path.join(cfg.DATA.DATA_ROOT_PATH, f"v{cfg.DATA.VERSION}")

        if isinstance(cfg.DATA.FEAT_DIR, list):
            # Must be extracted videomae features
            self.features_path = [
                os.path.join(cfg.DATA.DATA_ROOT_PATH, feat_dir)
                for feat_dir in cfg.DATA.FEAT_DIR]
        else:
            self.features_path = os.path.join(self.root_path, cfg.DATA.FEAT_DIR)

        anno_path = os.path.join(self.root_path, "annotations")
        taxonomy_path = os.path.join(anno_path, 'fho_lta_taxonomy.json')
        self.taxonomy = utils.load_json(taxonomy_path)
        self.action_taxonomy = utils.load_json(os.path.join(anno_path, "action_taxonomy.json"))
        action_freq = [val["freq"] for val in self.action_taxonomy.values()]
        self.action_freq = torch.tensor(action_freq, dtype=torch.float32)

        # Get verb-noun cooccurrence matrix
        verb_noun_cooccurrences = {(0, 0): 0}
        verb_noun_cooccurrences.update({
            (int(key.split(",")[0]) + 1, int(key.split(",")[1]) + 1):
                val["freq"] for key, val in self.action_taxonomy.items()
        })
        self.verb_noun_cooccurrences = utils.get_action_matrix(
            frozenset(verb_noun_cooccurrences.items()), init=0).float()

        self.verb_noun_to_action = {(0, 0): 0}
        self.verb_noun_to_action.update({
            (int(key.split(",")[0]) + 1, int(key.split(",")[1]) + 1):
                val["action_label"] + 1 for key, val in self.action_taxonomy.items()})

        self.feature_step_in_sec = self.feature_stride / 30
        self.long_memory_length = int(
            cfg.DATA.LONG_MEMORY_LENGTH // self.feature_step_in_sec)
        self.work_memory_length = int(
            (cfg.DATA.TAU_O - cfg.DATA.LONG_MEMORY_LENGTH) // self.feature_step_in_sec)

        # Important:  We include the background class as 0
        self.num_classes = {
            "verb": len(self.taxonomy["verbs"]) + 1,
            "noun": len(self.taxonomy["nouns"]) + 1,
            "action": len(self.action_taxonomy) + 1
        }

        self.class_mappings = self._get_class_mappings()

        mode = 'test_unannotated' if mode == 'test' else mode
        self.gt_path = os.path.join(anno_path, f'fho_lta_{mode}.json')

        # Load video lists if possible.
        sorted_anno_path = os.path.join(anno_path, f'fho_lta_{mode}_sorted.json')
        vid_list_path = os.path.join(
            anno_path, f'fho_lta_{mode}_{self.n_obs}_{self.n_future}.json')
        if not os.path.exists(vid_list_path):
            self.sorted_anno, self.vid_list = self.prepare_video_list()
            logger.info("Saving annotations ...")
            utils.dump_json(self.sorted_anno, sorted_anno_path)
            utils.dump_json(self.vid_list, vid_list_path)
        else:
            self.sorted_anno = utils.load_json(sorted_anno_path)
            self.vid_list = utils.load_json(vid_list_path)
            logger.info("Loaded annotations.")

        self.data_cache = Manager().dict()  # shared dict for features

    def prepare_video_list(self):
        # modified from https://github.com/EGO4D/forecasting/blob/main/ego4d_forecasting/datasets/ptv_dataset_helper.py#L395
        entries = utils.load_json(self.gt_path)["clips"]

        # if entries do not have verb/noun labels (test set) then give dummy ones
        for entry in entries:
            if 'verb_label' not in entry:
                entry.update({'verb_label': -1, 'noun_label': -1})

        # group annotations by clip_uid
        annotations = collections.defaultdict(list)
        for entry in entries:
            annotations[entry['clip_uid']].append(entry)

        # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
        annotations = {
            clip_uid: sorted(annotations[clip_uid],
                             key=lambda x: x['action_idx'])
            for clip_uid in annotations
        }

        # add key "feature_clip_start_frame", added by Zeyun
        annotations = {
            clip_uid: [
                {
                    **item,
                    "feature_clip_start_frame": frame_to_feat(
                        item["action_clip_start_frame"], self.feature_stride),
                    "feature_clip_end_frame": frame_to_feat(
                        item["action_clip_end_frame"], self.feature_stride)
                }
                for item in clip_list
            ]
            for clip_uid, clip_list in annotations.items()
        }

        untrimmed_clip_annotations = []
        video_path_prefix = ""
        num_future_actions = self.n_future
        num_input_actions = self.n_obs
        for clip_uid, video_clips in tqdm(annotations.items(), desc="Preparing annotations ..."):
            video_path = os.path.join(video_path_prefix, f'{clip_uid}.mp4')
            if len(video_clips) <= 0:
                continue

            # Extract forecasting annotations from video clips.
            for i in range(
                    len(video_clips) - num_future_actions - num_input_actions + 1
            ):
                input_clips = copy.deepcopy(
                    video_clips[i: i + num_input_actions])
                forecast_clips = copy.deepcopy(
                    video_clips[
                    i + num_input_actions:
                    i + num_input_actions + num_future_actions
                    ]
                )
                untrimmed_clip_annotations.append(
                    (
                        video_path,
                        {
                            "input_clips": input_clips,
                            "forecast_clips": forecast_clips,
                        },
                    )
                )
        return annotations, untrimmed_clip_annotations

    def _get_class_mappings(self) -> dict[tuple[str, str], torch.Tensor]:
        num_verbs = self.num_classes["verb"]
        num_nouns = self.num_classes["noun"]
        num_actions = self.num_classes["action"]
        verb_in_action = torch.zeros((num_actions, num_verbs), dtype=torch.float)
        noun_in_action = torch.zeros((num_actions, num_nouns), dtype=torch.float)
        for verb_noun_str, action_dict in self.action_taxonomy.items():
            verb, noun = verb_noun_str.split(",")
            # Donot forget we include the background class 0
            verb, noun, action = int(verb) + 1, int(noun) + 1, action_dict["action_label"] + 1
            verb_in_action[action, verb] = 1.0
            noun_in_action[action, noun] = 1.0
        return {
            ('verb', 'action'): verb_in_action,
            ('noun', 'action'): noun_in_action
        }

    def __getitem__(self, idx):
        if self.mode == "test":
            return self._make_input_test(idx)
        return self._make_input(idx)

    def load_feats_and_labels(self, clip_uid):
        if isinstance(self.features_path, list):
            features = [torch.load(os.path.join(path, f"{clip_uid}.pt"))
                        for path in self.features_path]
            features = torch.cat(features, dim=-1)
        else:
            features = torch.load(os.path.join(self.features_path, f"{clip_uid}.pt"))
        if self.mode == "test":
            return features

        verbs = torch.load(os.path.join(self.root_path, "verb_anno_perframe", f"{clip_uid}.pt"))
        nouns = torch.load(os.path.join(self.root_path, "noun_anno_perframe", f"{clip_uid}.pt"))
        actions = torch.load(os.path.join(self.root_path, "action_anno_perframe", f"{clip_uid}.pt"))
        return features, actions, verbs, nouns

    def construct_future_labels(self, forecast_clips):
        nouns = [clip["noun_label"] for clip in forecast_clips]
        verbs = [clip["verb_label"] for clip in forecast_clips]
        actions = [self.action_taxonomy[f"{verb},{noun}"]["action_label"] for verb, noun in zip(verbs, nouns)]

        # Donot forget to include the background class 0
        nouns = F.one_hot(torch.tensor(nouns) + 1, self.num_classes["noun"]).float()
        verbs = F.one_hot(torch.tensor(verbs) + 1, self.num_classes["verb"]).float()
        actions = F.one_hot(torch.tensor(actions) + 1, self.num_classes["action"]).float()

        return actions, verbs, nouns

    def _pad_feats(self, feat, max_len):
        padding = max_len - len(feat)
        if padding > 0:
            pad = torch.zeros(padding, feat.size(-1), dtype=feat.dtype)
            feat = torch.cat((pad, feat), dim=0)
        return feat

    def _pad_labels(self, labels, max_len):
        padding = max_len - len(labels)
        if padding > 0:
            pad = torch.zeros(padding, labels.size(-1), dtype=labels.dtype)
            pad[: 0] = 1.  # background class
            labels = torch.cat((pad, labels), dim=0)
        return labels

    def _make_input(self, video_index):
        vid_file, info_dict = self.vid_list[video_index]
        input_clips = info_dict["input_clips"]
        forecast_clips = info_dict["forecast_clips"]

        clip_uid = vid_file.split(".")[0]
        if clip_uid not in self.data_cache:
            features, actions, verbs, nouns = self.load_feats_and_labels(clip_uid)
            # Since we include a background class as 0.
            actions, verbs, nouns = actions + 1, verbs + 1, nouns + 1
            self.data_cache[clip_uid] = (features, actions, verbs, nouns)
        else:
            features, actions, verbs, nouns = self.data_cache[clip_uid]

        action_anno = utils.to_one_hot(actions, self.num_classes["action"])
        verb_anno = utils.to_one_hot(verbs, self.num_classes["verb"])
        noun_anno = utils.to_one_hot(nouns, self.num_classes["noun"])

        work_end = max([clip["feature_clip_end_frame"] for clip in input_clips])
        work_start = max(0, work_end - self.work_memory_length)
        long_start = max(0, work_start - self.long_memory_length)

        # Get work memory and labels, pad if necessary
        work_feats = features[work_start:work_end]
        work_action = action_anno[work_start: work_end]
        work_verb = verb_anno[work_start: work_end]
        work_noun = noun_anno[work_start: work_end]
        if len(work_feats) < self.work_memory_length:
            work_feats = self._pad_feats(work_feats, self.work_memory_length)
            work_action = self._pad_labels(work_action, self.work_memory_length)
            work_verb = self._pad_labels(work_verb, self.work_memory_length)
            work_noun = self._pad_labels(work_noun, self.work_memory_length)

        # Get long memory, pad if necessary
        long_feats = features[long_start:work_start]
        if len(long_feats) < self.long_memory_length:
            long_feats = self._pad_feats(long_feats, self.long_memory_length)

        past_features = torch.cat((long_feats, work_feats), dim=0)
        future_act, future_verb, future_noun = self.construct_future_labels(forecast_clips)

        action_idx = input_clips[-1]["action_idx"]
        item = {
            "past_feats": past_features,
            "past_act": work_action,
            "past_verb": work_verb,
            "past_noun": work_noun,
            "future_act": future_act,
            "future_verb": future_verb,
            "future_noun": future_noun,
            "clip_uid_action_idx": f"{clip_uid}_{action_idx}",
        }
        return item

    def _make_input_test(self, video_index):
        vid_file, info_dict = self.vid_list[video_index]
        input_clips = info_dict["input_clips"]
        forecast_clips = info_dict["forecast_clips"]

        clip_uid = vid_file.split(".")[0]
        if clip_uid not in self.data_cache:
            features = self.load_feats_and_labels(clip_uid)
            self.data_cache[clip_uid] = features
        else:
            features = self.data_cache[clip_uid]

        work_end = max([clip["feature_clip_end_frame"] for clip in input_clips])
        work_start = max(0, work_end - self.work_memory_length)
        long_start = max(0, work_start - self.long_memory_length)

        # Get work memory and labels, pad if necessary
        work_feats = features[work_start:work_end]
        if len(work_feats) < self.work_memory_length:
            work_feats = self._pad_feats(work_feats, self.work_memory_length)

        # Get long memory, pad if necessary
        long_feats = features[long_start:work_start]
        if len(long_feats) < self.long_memory_length:
            long_feats = self._pad_feats(long_feats, self.long_memory_length)

        past_features = torch.cat((long_feats, work_feats), dim=0)

        action_idx = input_clips[-1]["action_idx"]
        item = {
            "past_feats": past_features,
            "clip_uid_action_idx": f"{clip_uid}_{action_idx}",
        }
        return item

    def __len__(self):
        return len(self.vid_list)
