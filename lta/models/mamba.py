import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
from functools import partial

from lta.config import Config
from lta.models.build import MODEL_REGISTRY
from lta.models.transformer import QueryDecoder
from lta.utils.ouput_target_structure import Prediction
from lta.models.classification_head import ClassificationHead


class MAMBA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        mamba_cfg = {'d_state': cfg.MODEL.D_STATE, 'd_conv': cfg.MODEL.D_CONV}
        self.layers = nn.ModuleList(
            [
                create_block(
                    cfg.MODEL.D_MODEL,
                    ssm_cfg=mamba_cfg,
                    rms_norm=True,
                    fused_add_norm=True,
                    layer_idx=i,
                )
                for i in range(cfg.MODEL.N_LAYER)
            ]
        )

        self.norm_f = RMSNorm(cfg.MODEL.D_MODEL)
        self.apply(partial(_init_weights, n_layer=cfg.MODEL.N_LAYER))

    def forward(self, hidden_states, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        # Set prenorm=False here since we don't need the residual
        fused_add_norm_fn = rms_norm_fn
        hidden_states = fused_add_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
        )
        return hidden_states

    @torch.inference_mode()
    def generate(self, inputs, max_length, inference_params=None):
        batch_size, seqlen_og = inputs.shape[:2]

        # cache for decoding
        if inference_params is None:
            inference_params = InferenceParams(
                max_seqlen=max_length, max_batch_size=batch_size)

        def should_stop(inference_params):
            if inference_params.seqlen_offset >= max_length:
                return True
            return False

        sequences = [inputs]
        while not should_stop(inference_params):
            out = self(sequences[-1], inference_params)
            sequences.append(out[:, -1:])
            inference_params.seqlen_offset += 1

        sequences = torch.cat(sequences[1:], dim=1)
        return sequences


@MODEL_REGISTRY.register()
class QueryMAMBA(nn.Module):
    def __init__(self, cfg: Config, num_classes: dict[str, int], dataset):
        super().__init__()
        self.cfg = cfg
        self.encoder = MAMBA(cfg)
        self.decoder = QueryDecoder(cfg)
        self.long_mem_len = int(
            cfg.DATA.LONG_MEMORY_LENGTH // cfg.DATA.PAST_STEP_IN_SEC)

        self.drop = cfg.MODEL.DROPOUT
        if self.drop > 0:
            self.dropout = nn.Dropout(self.drop)

        # Linear layers for dimension conversion
        self.input_dim_converter = nn.Identity()
        if cfg.MODEL.INPUT_DIM != cfg.MODEL.D_MODEL:
            self.input_dim_converter = nn.Linear(
                cfg.MODEL.INPUT_DIM, cfg.MODEL.D_MODEL)

        # Classification head
        self.head = ClassificationHead(cfg, num_classes, dataset)

    def forward(self, past, future=None) -> Prediction:
        # Encoder
        past = self.input_dim_converter(past)

        if self.drop > 0:
            past = self.dropout(past)

        past = self.encoder(past)

        # Split memory into long-term mem and working mem
        long_mem, work_mem = past[:, :self.long_mem_len], past[:, self.long_mem_len:]

        # Decoder
        future_pred = self.decoder(work_mem)

        # Classification
        out = self.head(work_mem, future_pred)
        return out
