# Inference-only adaptation of RoMa (Johan Edstedt, MIT).
# Modified for standalone GPDRNet; see README.md (Attribution and release license).
from typing import Literal
import torch
import torch.nn.functional as F


def native_torch_local_corr(
    feature0,
    feature1,
    warp,
    local_window,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode="bilinear",
    dtype=torch.float32,
):
    corr = torch.empty((B, K, h, w), device=device, dtype=dtype)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (warp[_, :, :, None] + local_window[:, None, None]).reshape(
                1, h, w * K, 2
            )
            window_feature = F.grid_sample(
                feature1[_ : _ + 1],
                local_window_coords,
                padding_mode=padding_mode,
                align_corners=False,
                mode=sample_mode,
            )
            window_feature = window_feature.reshape(c, h, w, K)
        corr[_] = (feature0[_, ..., None] / c ** 0.5 * window_feature).sum(dim=0).permute(2, 0, 1)
    return (corr, None)


def local_correlation(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    local_radius: int,
    warp: torch.Tensor,
    *,
    use_custom_corr: bool,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear"
):
    r = local_radius
    K = (2 * r + 1) ** 2
    (B, c, h, w) = feature0.size()
    warp = warp.permute(0, 2, 3, 1)
    device = feature0.device
    dtype = feature0.dtype
    local_window = torch.meshgrid(
        [
            torch.linspace(-2 * local_radius / h, 2 * local_radius / h, 2 * r + 1, device=device),
            torch.linspace(-2 * local_radius / w, 2 * local_radius / w, 2 * r + 1, device=device),
        ],
        indexing="ij",
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * r + 1, 2 * r + 1, 2)
        .reshape(1, K, 2)
    )
    (corr, corr_coords) = native_torch_local_corr(
        feature0,
        feature1,
        warp,
        local_window,
        B,
        K,
        c,
        r,
        h,
        w,
        device,
        padding_mode,
        sample_mode,
        dtype,
    )
    return corr
