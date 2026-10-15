"""GPDRNet matching followed by the separate Learning-based LDAPP module."""
from importlib import import_module
from pathlib import Path

import torch
from torch import nn

from .network import GPDRNet

# The requested filename contains a hyphen and a space, so use string-based import.
LearningBasedLDAPP = import_module(".learning-based ldapp", __package__).LearningBasedLDAPP

BACKBONE_PREFIX = "matcher.encoder.dinov2."


class RegistrationPipeline(nn.Module):
    """Compose GPDRNet and Learning-based LDAPP without conflating their roles."""

    def __init__(self):
        super().__init__()
        self.gpdrnet = GPDRNet()
        self.ldapp = LearningBasedLDAPP()
        self.requires_grad_(False)
        self.eval()

    def load_weights(self, weights, backbone, ldapp):
        """Load three raw state dictionaries; no missing learned parameters allowed.

        registration.pth contains GPDRNet except DINOv2; ldapp.pth contains LDAPP.
        Each dictionary uses its own module's keys, without a pipeline prefix.
        The backbone dictionary uses original, unprefixed DINOv2 state keys.
        DINOv2 remains a registered runtime submodule but is stored separately.
        """
        expected = set(self.gpdrnet.state_dict())
        backbone_keys = {key for key in expected if key.startswith(BACKBONE_PREFIX)}
        missing = expected - backbone_keys - set(weights)
        unexpected = set(weights) - (expected - backbone_keys)
        if missing or unexpected:
            raise ValueError(
                f"Invalid registration.pth keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.gpdrnet.matcher.encoder.dinov2.load_state_dict(backbone, strict=True)
        result = self.gpdrnet.load_state_dict(weights, strict=False)
        if set(result.missing_keys) != backbone_keys or result.unexpected_keys:
            raise RuntimeError("Unexpected missing/unrecognized model parameters")
        self.ldapp.load_state_dict(ldapp, strict=True)
        return self

    @torch.inference_mode()
    def forward(
        self, sar, optical, valid_sar=None, valid_optical=None, *, blend="smooth", seed=20261002,
        cycle_threshold_px=3.0,
    ):
        if self.training:
            raise RuntimeError("This pipeline is inference-only; call model.eval().")
        if blend not in ("smooth", "safe", "hard"):
            raise ValueError("blend must be smooth, safe, or hard")
        masks = []
        for mask in (valid_sar, valid_optical):
            if mask is None:
                mask = torch.ones((1, 1, 512, 512), dtype=torch.bool, device=sar.device)
            if mask.shape != (1, 1, 512, 512):
                raise ValueError("Expected observed image masks [1,1,512,512]")
            masks.append(mask.to(device=sar.device, dtype=torch.bool))
        observed = torch.cat(masks)
        prediction = self.gpdrnet(sar, optical)
        with torch.autocast("cuda", enabled=False):
            result = self.ldapp(
                prediction["coarse_flow"],
                prediction["flow"],
                prediction["confidence_logits"],
                observed,
                prediction["correlation_inputs"],
                blend=blend,
                seed=seed,
                cycle_threshold_px=cycle_threshold_px,
            )
        result.update(
            coarse_flow=prediction["coarse_flow"],
            raw_flow=prediction["flow"],
            confidence_logits=prediction["confidence_logits"],
        )
        return result


def load_model(
    weights: str | Path, backbone: str | Path, ldapp_weights: str | Path, device="cuda"
) -> RegistrationPipeline:
    """Load separate GPDRNet, DINOv2, and Learning-based LDAPP checkpoints offline."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU with native bfloat16 support is required")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("The checkpoint uses native bfloat16 inference")
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    backbone_state = torch.load(backbone, map_location="cpu", weights_only=True, mmap=True)
    ldapp_state = torch.load(ldapp_weights, map_location="cpu", weights_only=True, mmap=True)
    model = RegistrationPipeline()
    model.load_weights(state, backbone_state, ldapp_state)
    return model.to(device).eval()
