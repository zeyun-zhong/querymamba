from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@dataclass
class Prediction:
    past_feats: Optional[Tensor] = None
    future_feats: Optional[Tensor] = None
    past_actions: Optional[Tensor] = None
    future_actions: Optional[Tensor] = None
    past_verbs: Optional[Tensor] = None
    past_nouns: Optional[Tensor] = None
    future_verbs: Optional[Tensor] = None
    future_nouns: Optional[Tensor] = None


@dataclass
class Target:
    past_feats: Optional[Tensor] = None
    future_feats: Optional[Tensor] = None
    past_actions: Optional[Tensor] = None
    future_actions: Optional[Tensor] = None
    past_verbs: Optional[Tensor] = None
    past_nouns: Optional[Tensor] = None
    future_verbs: Optional[Tensor] = None
    future_nouns: Optional[Tensor] = None
    mixup_enabled: Optional[bool] = False
    vid_name: Optional[str] = None
    work_indices: Optional[Tensor] = None
    num_frames: Optional[int] = None