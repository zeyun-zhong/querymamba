from dataclasses import dataclass, field

import torch
from nestconfig import NestConfig
from typing import Union


@dataclass
class DataConfig:
    DATA_ROOT_PATH = "/home/zhong/Documents/datasets"  # path where all datasets are saved
    DATASET_CLASS: str = "Ego4D"
    FEAT_DIR = "videomae_vitl_ego4d_verb_clips"

    # Short-term specific
    TAU_O: float = 69.  # length of observation, in seconds (working mem + long mem)
    PAST_STEP_IN_SEC: float = 0.53333333
    LONG_MEMORY_LENGTH: float = 64.  # secs of long-term memory, if fps is 4 -> 256 sequence length

    # Ego4D related
    VERSION: int = 1
    N_OBS: int = 8  # num of observed segments, default 8 in ego4d


@dataclass
class ModelConfig:
    MODEL_CLASS: str = "QueryMAMBA"
    CRITERION_CLASS: str = "Criterion_Ego4D"
    INPUT_DIM: int = 1024
    D_MODEL: int = 512
    N_LAYER: int = 2
    N_DEC_LAYER: int = 2
    D_STATE: int = 16
    D_CONV: int = 4
    SHARE_CLASSIFIER: bool = False
    IGNORE_INDEX: int = -1  # class that does not contribute to the loss

    PAST_CLS: bool = True

    # Action, Verb, Noun Classification
    ACTION_CLS: bool = True
    VERB_CLS: bool = False
    NOUN_CLS: bool = False

    D_FFN: int = D_MODEL * 4
    D_ATT: int = D_MODEL
    HEAD_SIZE: int = 64
    DROPOUT: float = 0
    DROP_CLS: float = 0.

    # For querydecoder
    PRENORM: bool = False
    N_QUERIES: int = 1


@dataclass
class TrainConfig:
    ENABLE: bool = True
    CKPT_PATH: str = None
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    OPTIMIZER: str = "sgd"
    WEIGHT_DECAY: float = 0.
    SCHEDULER: str = "cosine"
    EPOCHS: int = 50
    WARMUP_STEPS: int = 5
    LR: float = 0.001
    MIN_LR: float = 1e-7
    GRADIENT_CLIPPING: Union[float, None] = None
    USE_MIXUP: bool = False
    SAVE_MODEL: bool = True

    # Loss functions
    ACTION2VERBNOUN: bool = False
    VERBNOUN2ACTION: bool = False


@dataclass
class ValConfig:
    ENABLE: bool = True
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    EVALUATE_EVERY: int = 1


@dataclass
class TestConfig:
    ENABLE: bool = False
    CKPT_PATH: str = None


@dataclass
class Config(NestConfig):
    SEED = 42
    PRIMARY_METRIC = "val/action_ed_marginalize"
    NOTE = None   # some notes of the experiment
    USE_WANDB = False  # whether to use wandb to visualize logs
    LOG_LEVEL = 'info'  # info or debug
    WANDB_PROJECT = None
    METRIC_DESCENDING: bool = False
    DTYPE: str = "float32"

    MODEL: ModelConfig = field(default_factory=ModelConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    VAL: ValConfig = field(default_factory=ValConfig)
    TEST: TestConfig = field(default_factory=TestConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
