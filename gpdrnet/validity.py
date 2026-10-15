"""Observed visibility and bidirectional geometry, never ground-truth cropping.

This rejects unreliable output pixels; it does not change correspondences or
certify their accuracy. No global affine footprint is imposed on elastic flow.
"""
import math

import torch
import torch.nn.functional as F

from .geometry import identity_grid, inside, sample


def _nonfolding_pixels(flow):
    """Reject every corner of a cell with a nonpositive bilinear Jacobian."""
    height, width = flow.shape[-2:]
    xy = flow.permute(0, 2, 3, 1) * flow.new_tensor((width / 2, height / 2))
    top = xy[:, :-1, 1:] - xy[:, :-1, :-1]
    bottom = xy[:, 1:, 1:] - xy[:, 1:, :-1]
    left = xy[:, 1:, :-1] - xy[:, :-1, :-1]
    right = xy[:, 1:, 1:] - xy[:, :-1, 1:]
    determinants = torch.stack([
        dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        for dx, dy in ((top, left), (top, right), (bottom, left), (bottom, right))
    ])
    bad_cell = (determinants <= 0).any(0) | ~torch.isfinite(determinants).all(0)
    bad_pixel = torch.zeros_like(flow[:, :1], dtype=torch.bool)
    for padding in ((0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)):
        bad_pixel |= F.pad(bad_cell[:, None], padding)
    return ~bad_pixel


@torch.no_grad()
def registration_validity(flow, observed, *, affine_flow=None, cycle_threshold_px=3.0):
    """Return masks for two opposite absolute fields on the same image grid.

The round-trip threshold is measured on this grid (512 pixels in the release),
not at native280 confidence resolution. It is a heuristic, not calibrated
    matching confidence. Both endpoints must be observed and locally nonfolding.

    boundary_valid is the conservative default: reject only where the predicted
    affine footprint AND the dense reliability test disagree with visibility.
    Consistent nonrigid matches outside that footprint remain valid. Without an
    accepted affine fit it falls back to observed visibility, not a guessed crop.
"""
    if flow.ndim != 4 or flow.shape[:2] != (2, 2) or min(flow.shape[-2:]) < 2:
        raise ValueError("Expected two opposite fields [2,2,H,W], H,W >= 2")
    if observed.shape != (2, 1, *flow.shape[-2:]):
        raise ValueError("Expected observed masks [2,1,H,W] on the flow grid")
    if not math.isfinite(cycle_threshold_px) or cycle_threshold_px <= 0:
        raise ValueError("cycle_threshold_px must be finite and positive")
    if not torch.isfinite(flow).all():
        raise ValueError("Flow must be finite")
    flow = flow.float()
    observed = observed.to(device=flow.device, dtype=torch.bool)
    height, width = flow.shape[-2:]
    grid = flow.permute(0, 2, 3, 1)
    identity = identity_grid(height, width, device=flow.device).permute(0, 3, 1, 2)
    reverse = flow.flip(0)
    visible = observed & inside(grid, height, width)[:, None]
    visible &= sample(observed.flip(0).float(), grid) > 0.999
    local_valid = visible & _nonfolding_pixels(flow)
    reverse_valid = sample(local_valid.flip(0).float(), grid) > 0.999
    returned = sample(reverse, grid)
    scale = flow.new_tensor((width / 2, height / 2))[None, :, None, None]
    cycle_error = ((returned - identity) * scale).norm(dim=1, keepdim=True)
    valid = local_valid & reverse_valid & (cycle_error <= cycle_threshold_px)
    boundary_valid = visible.clone()
    if affine_flow is not None:
        if affine_flow.shape != flow.shape or not torch.isfinite(affine_flow).all():
            raise ValueError("Expected finite affine fields with the same shape as flow")
        affine_grid = affine_flow.to(device=flow.device, dtype=torch.float32).permute(0, 2, 3, 1)
        affine_visible = inside(affine_grid, height, width)[:, None]
        affine_visible &= sample(observed.flip(0).float(), affine_grid) > 0.999
        boundary_valid &= affine_visible | valid
    return {"valid": valid, "boundary_valid": boundary_valid,
            "observed_valid": visible, "cycle_error_px": cycle_error}
