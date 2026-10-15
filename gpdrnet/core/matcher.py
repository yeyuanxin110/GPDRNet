# Inference-only adaptation of RoMa (Johan Edstedt, MIT).
# Modified for standalone GPDRNet; see README.md (Attribution and release license).
import sys
import math
from warnings import warn
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .utils import get_autocast_params, cls_to_flow_refine
from .local_correlation import local_correlation


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb=None,
        displacement_emb_dim=None,
        local_corr_radius=None,
        corr_in_other=None,
        no_im_B_fm=False,
        amp=False,
        concat_logits=False,
        use_bias_block_1=True,
        use_cosine_corr=False,
        disable_local_corr_grad=False,
        is_classifier=False,
        sample_mode="bilinear",
        norm_type=nn.BatchNorm2d,
        bn_momentum=0.1,
        amp_dtype=torch.float16,
        use_custom_corr=False,
    ):
        super().__init__()
        if sys.platform != "linux":
            warn(
                "Local correlation is not supported on non-Linux platforms, setting use_custom_corr to False"
            )
            use_custom_corr = False
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias=use_bias_block_1
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim, hidden_dim, dw=dw, kernel_size=kernel_size, norm_type=norm_type
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        self.use_custom_corr = use_custom_corr

    def create_block(
        self, in_dim, out_dim, dw=False, kernel_size=5, bias=True, norm_type=nn.BatchNorm2d
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim, momentum=self.bn_momentum)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, warp, scale_factor=1, logits=None):
        (b, c, hs, ws) = x.shape
        (autocast_device, autocast_enabled, autocast_dtype) = get_autocast_params(
            x.device, enabled=self.amp, dtype=self.amp_dtype
        )
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            x_hat = F.grid_sample(
                y, warp.permute(0, 2, 3, 1), align_corners=False, mode=self.sample_mode
            )
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                    (
                        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                    ),
                    indexing="ij",
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = warp - im_A_coords
                emb_in_displacement = self.disp_emb(40 / 32 * scale_factor * in_displacement)
                if self.local_corr_radius:
                    if self.corr_in_other:
                        local_corr = local_correlation(
                            x,
                            y,
                            self.local_corr_radius,
                            warp,
                            sample_mode=self.sample_mode,
                            use_custom_corr=self.use_custom_corr,
                        )
                    else:
                        raise NotImplementedError("Local corr in own frame should not be used.")
                    if self.no_im_B_fm:
                        x_hat = torch.zeros_like(x)
                    d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
                else:
                    d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            channel_d = d.shape[1]
            channel_block1 = self.block1[0].in_channels
            if channel_d != channel_block1:
                d = F.pad(d, (0, 0, 0, 0, 0, channel_block1 - channel_d))
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        (displacement, certainty) = (d[:, :-1], d[:, -1:])
        return (displacement, certainty)


class CosKernel(nn.Module):
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-06):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features=False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        (b, c, h, w) = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing="ij",
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[None].expand(
            b, h, w, 2
        )
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        (b, c, h1, w1) = x.shape
        (b, c, h2, w2) = y.shape
        f = self.get_pos_enc(y)
        (b, d, h2, w2) = f.shape
        (x, y, f) = (self.reshape(x.float()), self.reshape(y.float()), self.reshape(f.float()))
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        L_t = torch.linalg.cholesky(K_yy + sigma_noise)
        pos_emb = torch.cholesky_solve(f.reshape(b, h2 * w2, d), L_t, upper=False)
        mu_x = K_xy @ pos_emb
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        gp_feats = mu_x
        return gp_feats


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_decoder,
        gps,
        proj,
        conv_refiner,
        detach=False,
        scales="all",
        pos_embeddings=None,
        num_refinement_steps_per_scale=1,
        warp_noise_std=0.0,
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
        flow_upsample_mode="bilinear",
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.upsample_conv_refiner = None
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing="ij",
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[None].expand(
            b, h, w, 2
        )
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def forward(
        self, f1, f2, upsample=False, flow=None, certainty=None, scale_factor=1, coarse_only=False
    ):
        coarse_scales = self.embedding_decoder.scales()
        if coarse_only and upsample:
            raise ValueError("coarse_only and upsample cannot both be enabled")
        all_scales = (
            [str(scale) for scale in coarse_scales]
            if coarse_only
            else self.scales
            if not upsample
            else ["8", "4", "2", "1"]
        )
        conv_refiner = (
            self.upsample_conv_refiner
            if upsample and self.upsample_conv_refiner is not None
            else self.conv_refiner
        )
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        (h, w) = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b,
            self.embedding_decoder.hidden_dim,
            *sizes[coarsest_scale],
            device=f1[coarsest_scale].device
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                flow, size=sizes[coarsest_scale], align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
                certainty, size=sizes[coarsest_scale], align_corners=False, mode="bilinear"
            )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            (f1_s, f2_s) = (f1[ins], f2[ins])
            if new_scale in self.proj:
                (autocast_device, autocast_enabled, autocast_dtype) = get_autocast_params(
                    f1_s.device, str(f1_s) == "cuda", self.amp_dtype
                )
                with torch.autocast(
                    autocast_device, enabled=autocast_enabled, dtype=autocast_dtype
                ):
                    if not autocast_enabled:
                        (f1_s, f2_s) = (f1_s.to(torch.float32), f2_s.to(torch.float32))
                    (f1_s, f2_s) = (self.proj[new_scale](f1_s), self.proj[new_scale](f2_s))
            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                (gm_warp_or_cls, certainty, old_stuff) = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(gm_warp_or_cls).permute(0, 3, 1, 2)
                else:
                    flow = gm_warp_or_cls.detach()
            if new_scale in conv_refiner:
                (delta_flow, delta_certainty) = conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor=scale_factor, logits=certainty
                )
                displacement = ins * torch.stack(
                    (
                        delta_flow[:, 0].float() / (self.refine_init * w),
                        delta_flow[:, 1].float() / (self.refine_init * h),
                    ),
                    dim=1,
                )
                flow = flow + displacement
                certainty = certainty + delta_certainty
            corresps[ins].update({"certainty": certainty, "flow": flow})
            if new_scale != "1" and (not coarse_only):
                flow = F.interpolate(flow, size=sizes[ins // 2], mode=self.flow_upsample_mode)
                certainty = F.interpolate(
                    certainty, size=sizes[ins // 2], mode=self.flow_upsample_mode
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
        return corresps
