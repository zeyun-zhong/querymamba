import torch
from torch import Tensor

from lta.config import Config
from lta.utils import accuracy, Prediction, Target
from lta.utils.metrics import edit_distance
from lta.criterion.build import Criterion_REGISTRY
from lta.datasets import Ego4D, action2verbnoun, verbnoun2action, get_action_matrix
from lta.criterion.loss import MultipCrossEntropyEqualizedLoss, MultipCrossEntropyLoss


@Criterion_REGISTRY.register()
class Criterion_Ego4D:
    def __init__(self, cfg: Config, dataset: Ego4D):
        ignore_index = cfg.MODEL.IGNORE_INDEX
        self.action_cls = MultipCrossEntropyEqualizedLoss(
            ignore_index=ignore_index, freq_info=dataset.action_freq)
        self.verb_cls = MultipCrossEntropyLoss(ignore_index=ignore_index)
        self.noun_cls = MultipCrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.dataset = dataset
        self.cfg = cfg

    def __call__(self, pred: Prediction, target: Target, is_training=True) -> (Tensor, dict):
        loss, loss_dict = 0, {}

        # Past loss
        if pred.past_verbs is not None:
            past_verb = self.verb_cls(pred.past_verbs, target.past_verbs)
            past_noun = self.noun_cls(pred.past_nouns, target.past_nouns)
            loss += past_verb + past_noun
            loss_dict["past_verb_loss"] = past_verb.item()
            loss_dict["past_noun_loss"] = past_noun.item()

        # Future loss
        future_verb = self.verb_cls(pred.future_verbs, target.future_verbs)
        future_noun = self.noun_cls(pred.future_nouns, target.future_nouns)

        loss += future_verb + future_noun
        loss_dict["future_verb_loss"] = future_verb.item()
        loss_dict["future_noun_loss"] = future_noun.item()

        if pred.past_actions is not None:
            past_cls = self.action_cls(pred.past_actions, target.past_actions)
            loss += past_cls
            loss_dict["past_action_loss"] = past_cls.item()

        if pred.future_actions is not None:
            future_cls = self.action_cls(pred.future_actions, target.future_actions)
            loss += future_cls
            loss_dict["future_action_loss"] = future_cls.item()

        loss_dict["loss"] = loss.item()

        if not is_training:
            verb_notice_index = [i for i in range(target.future_verbs.shape[-1])
                                 if i != self.ignore_index]
            noun_notice_index = [i for i in range(target.future_nouns.shape[-1])
                                 if i != self.ignore_index]
            verb_target = target.future_verbs[..., verb_notice_index].argmax(dim=-1)
            noun_target = target.future_nouns[..., noun_notice_index].argmax(dim=-1)

            verb_ed = edit_distance(pred.future_verbs[..., verb_notice_index], verb_target)
            noun_ed = edit_distance(pred.future_nouns[..., noun_notice_index], noun_target)

            # Transform action predictions to verb and noun
            action_marginalize = verbnoun2action(
                pred.future_verbs, pred.future_nouns,
                self.dataset.verb_noun_to_action)
            notice_index = [i for i in range(target.past_actions.shape[-1]) if
                            i != self.ignore_index]
            action_target = target.future_actions[..., notice_index].argmax(dim=-1)
            action_ed_marginalize = edit_distance(
                action_marginalize[..., notice_index], action_target)

            loss_dict.update({
                "verb_ed": verb_ed,
                "noun_ed": noun_ed,
                "action_ed_marginalize": action_ed_marginalize,
            })

            # Action
            if pred.future_actions is not None:
                action_ed = edit_distance(
                    pred.future_actions[..., notice_index], action_target)
                # Transform action predictions to verb and noun
                verb_marginalize, noun_marginalize = action2verbnoun(pred.future_actions, self.dataset.class_mappings)
                verb_ed_marginalize = edit_distance(verb_marginalize[..., verb_notice_index], verb_target)
                noun_ed_marginalize = edit_distance(noun_marginalize[..., noun_notice_index], noun_target)

                loss_dict.update({
                    "action_ed": action_ed,
                    "verb_ed_marginalize": verb_ed_marginalize,
                    "noun_ed_marginalize": noun_ed_marginalize,
                })

        return loss, loss_dict
