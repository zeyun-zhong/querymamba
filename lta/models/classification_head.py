import torch.nn as nn
import torch

from lta.config import Config
from lta.utils.ouput_target_structure import Prediction


class ClassificationHead(nn.Module):
    def __init__(self, cfg: Config, num_classes, dataset):
        super().__init__()
        self.cfg = cfg

        self.drop_cls = cfg.MODEL.DROP_CLS
        if self.drop_cls > 0:
            self.dropout_cls = nn.Dropout(self.drop_cls)

        if cfg.MODEL.PAST_CLS:
            if cfg.MODEL.ACTION_CLS:
                self.classifier_past = nn.Linear(cfg.MODEL.D_MODEL, num_classes["action"])

            if cfg.MODEL.VERB_CLS:
                self.classifier_past_verb = nn.Linear(cfg.MODEL.D_MODEL, num_classes["verb"])

            if cfg.MODEL.NOUN_CLS:
                self.classifier_past_noun = nn.Linear(cfg.MODEL.D_MODEL, num_classes["noun"])

        if cfg.MODEL.SHARE_CLASSIFIER:
            assert cfg.MODEL.PAST_CLS, \
                "Classifier cannot be shared because past classification is not enabled."
            if cfg.MODEL.ACTION_CLS:
                self.classifier_future = self.classifier_past
            if cfg.MODEL.VERB_CLS:
                self.classifier_future_verb = self.classifier_past_verb
            if cfg.MODEL.NOUN_CLS:
                self.classifier_future_noun = self.classifier_past_noun
        else:
            if cfg.MODEL.ACTION_CLS:
                self.classifier_future = nn.Linear(cfg.MODEL.D_MODEL, num_classes["action"])
            if cfg.MODEL.VERB_CLS:
                self.classifier_future_verb = nn.Linear(cfg.MODEL.D_MODEL, num_classes["verb"])
            if cfg.MODEL.NOUN_CLS:
                self.classifier_future_noun = nn.Linear(cfg.MODEL.D_MODEL, num_classes["noun"])

    def forward(self, work_mem, future_pred) -> Prediction:
        if self.drop_cls > 0:
            work_mem = self.dropout_cls(work_mem)
            future_pred = self.dropout_cls(future_pred)

        past_actions, future_actions = None, None
        if hasattr(self, "classifier_past"):
            past_actions = self.classifier_past(work_mem)

        if hasattr(self, "classifier_future"):
            future_actions = self.classifier_future(future_pred)

        past_verbs, past_nouns, future_verbs, future_nouns = None, None, None, None
        if hasattr(self, "classifier_past_verb"):
            past_verbs = self.classifier_past_verb(work_mem)

        if hasattr(self, "classifier_past_noun"):
            past_nouns = self.classifier_past_noun(work_mem)

        if hasattr(self, "classifier_future_verb"):
            future_verbs = self.classifier_future_verb(future_pred)

        if hasattr(self, "classifier_future_noun"):
            future_nouns = self.classifier_future_noun(future_pred)

        out = Prediction(
            past_feats=work_mem,
            future_feats=future_pred,
            past_actions=past_actions,
            future_actions=future_actions,
            past_verbs=past_verbs,
            future_verbs=future_verbs,
            past_nouns=past_nouns,
            future_nouns=future_nouns,
        )

        return out
