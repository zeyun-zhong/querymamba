from fvcore.common.registry import Registry

from lta.config import Config


Criterion_REGISTRY = Registry("Criterion")
Criterion_REGISTRY.__doc__ = """
Registry for criterions.

The registered object will be called with `obj(**kwargs)`.
The call should return a criterion object.
"""


def build_criterion(cfg: Config, dataset):
    """
    Build a model, defined by `cfg.MODEL.MODEL_CLASS`.
    """
    criterion = Criterion_REGISTRY.get(cfg.MODEL.CRITERION_CLASS)(cfg, dataset)
    return criterion