"""GPDRNet: coarse/fine matching at 280, confidence, and shared512 refinement."""
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import make_layers, cfgs

from .core.feature_enhancement import PairFeatureEnhancer
from .core.matcher import ConvRefiner, CosKernel, Decoder, GP
from .core.transformer import Block, MemEffAttention, TransformerDecoder, vit_large


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        # Same feature layers as vgg19_bn(...).features[:40], without allocating
        # its unused classification head or downloading classification weights.
        self.layers = nn.ModuleList(make_layers(cfgs["E"], batch_norm=True)[:40])

    def forward(self, x):
        features, scale = {}, 1
        with torch.autocast("cuda", dtype=torch.float16):
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    features[scale] = x
                    scale *= 2
                x = layer(x)
        return features


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = VGG19()
        self.dinov2 = vit_large(
            img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0
        ).to(torch.bfloat16)

    def forward(self, images, upsample=False):
        features = self.cnn(images)
        if not upsample:
            height, width = images.shape[-2:]
            padded = F.pad(images, (0, (-width) % 14, 0, (-height) % 14))
            tokens = self.dinov2.forward_features(padded.to(torch.bfloat16))["x_norm_patchtokens"]
            features[16] = tokens.permute(0, 2, 1).reshape(
                len(images), 1024, padded.shape[-2] // 14, padded.shape[-1] // 14
            )
        return features


class DenseMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        coordinate_decoder = TransformerDecoder(
            nn.Sequential(*[Block(1024, 8, attn_class=MemEffAttention) for _ in range(5)]),
            1024,
            64 ** 2 + 1,
            is_classifier=True,
            amp=True,
            pos_enc=False,
        )
        refiner = partial(
            ConvRefiner,
            kernel_size=5,
            dw=True,
            hidden_blocks=8,
            displacement_emb="linear",
            corr_in_other=True,
            amp=True,
            disable_local_corr_grad=True,
            bn_momentum=0.01,
            use_custom_corr=False,
        )
        refiners = nn.ModuleDict(
            {
                "16": refiner(1377, 1377, 3, displacement_emb_dim=128, local_corr_radius=7),
                "8": refiner(1137, 1137, 3, displacement_emb_dim=64, local_corr_radius=3),
                "4": refiner(569, 569, 3, displacement_emb_dim=32, local_corr_radius=2),
                "2": refiner(144, 144, 3, displacement_emb_dim=16),
                "1": refiner(24, 24, 3, displacement_emb_dim=6),
            }
        )
        projections = nn.ModuleDict(
            {
                str(s): nn.Sequential(nn.Conv2d(i, o, 1), nn.BatchNorm2d(o))
                for s, i, o in (
                    (16, 1024, 512),
                    (8, 512, 512),
                    (4, 256, 256),
                    (2, 128, 64),
                    (1, 64, 9),
                )
            }
        )
        self.decoder = Decoder(
            coordinate_decoder,
            nn.ModuleDict({"16": GP(CosKernel, T=0.2, gp_dim=512, basis="fourier", no_cov=True)}),
            projections,
            refiners,
            detach=True,
            scales=["16", "8", "4", "2", "1"],
        )
        self.feature_enhancer = PairFeatureEnhancer(
            {16: 1024, 8: 512, 4: 256, 2: 128},
            enhance_scales=[8, 4, 2],
            hidden_dim=128,
            num_layers=5,
            ffn_hidden_dim=512,
            window_size=7,
            outer_residual_init=0.01,
        )

    def forward(self, sar, optical, coarse=None):
        upsample = coarse is not None
        source = self.feature_enhancer(self.encoder(torch.cat((sar, optical)), upsample=upsample))
        target = {s: torch.cat((f.chunk(2)[1], f.chunk(2)[0])) for s, f in source.items()}
        return self.decoder(
            source,
            target,
            upsample=upsample,
            scale_factor=(512 if upsample else 280) / 560,
            **(coarse or {}),
        )


class ScaleReadout(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Conv2d(channels, 3, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, hidden):
        hidden = hidden.float()
        return self.linear(hidden) + self.residual(hidden)


class ConfidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.readouts = nn.ModuleDict(
            {str(s): ScaleReadout(c) for s, c in ((8, 1137), (4, 569), (2, 144), (1, 24))}
        )

    def forward(self, features, coarse_certainty):
        previous = coarse_certainty.float()
        previous = torch.cat((previous, torch.zeros_like(previous), torch.zeros_like(previous)), 1)
        for scale in (8, 4, 2, 1):
            delta = self.readouts[str(scale)](features[scale])
            previous = (
                F.interpolate(previous, delta.shape[-2:], mode="bilinear", align_corners=False)
                + delta
            )
        return previous


class GPDRNet(nn.Module):
    """Input A is SAR and B is optical; batch size is exactly one pair.

    Inputs: ImageNet-normalized float32 RGB tensors [1,3,512,512].
    Output flow[0] samples optical onto SAR; flow[1] samples SAR onto optical.
    All flows contain absolute normalized x,y coordinates (align_corners=False).
    This network stops at refined512 matching. Learning-based LDAPP is a
    separate postprocessing module, applied by RegistrationPipeline.
    """

    def __init__(self):
        super().__init__()
        self.matcher = DenseMatcher()
        self.confidence = ConfidenceHead()
        self._hidden, self._correlation = {}, {}
        for scale in (8, 4, 2, 1):
            self.matcher.decoder.conv_refiner[str(scale)].out_conv.register_forward_pre_hook(
                lambda module, inputs, scale=scale: self._hidden.__setitem__(scale, inputs[0])
            )
        for scale in (8, 4):
            self.matcher.decoder.conv_refiner[str(scale)].register_forward_pre_hook(
                lambda module, inputs, scale=scale: self._correlation.__setitem__(
                    scale, (inputs[0], inputs[1])
                )
            )
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, sar, optical):
        if self.training:
            raise RuntimeError("This is an inference-only network; call model.eval().")
        for image in (sar, optical):
            if image.shape != (1, 3, 512, 512) or image.dtype != torch.float32:
                raise ValueError("Expected float32 images [1,3,512,512]")
            if image.device.type != "cuda" or not torch.isfinite(image).all():
                raise ValueError("Finite CUDA image tensors are required")
        if sar.device != optical.device:
            raise ValueError("Both images must be on the same device")
        self._hidden.clear()
        self._correlation.clear()
        try:
            # Keep confidence prediction in FP32.
            with torch.autocast("cuda", enabled=False):
                native = [
                    F.interpolate(x, (280, 280), mode="bilinear", align_corners=False)
                    for x in (sar, optical)
                ]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = self.matcher(*native)
                coarse = {k: prediction[1][k] for k in ("flow", "certainty")}
                logits = self.confidence(self._hidden, prediction[16]["certainty"])
                self._hidden.clear()
                self._correlation.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    refined = self.matcher(sar, optical, coarse=coarse)[1]["flow"].float()
                if not all(torch.isfinite(x).all() for x in (coarse["flow"], refined, logits)):
                    raise FloatingPointError("Nonfinite matcher/confidence output")
                return dict(
                    coarse_flow=coarse["flow"].float(),
                    flow=refined,
                    confidence_logits=logits,
                    correlation_inputs=dict(self._correlation),
                )
        finally:
            self._hidden.clear()
            self._correlation.clear()
