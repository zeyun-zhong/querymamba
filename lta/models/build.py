from fvcore.common.registry import Registry

from lta.config import Config


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj(**kwargs)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg: Config, num_classes, dataset):
    """
    Build a model, defined by `cfg.MODEL.MODEL_CLASS`.
    """
    model = MODEL_REGISTRY.get(cfg.MODEL.MODEL_CLASS)(cfg, num_classes, dataset)
    return model
