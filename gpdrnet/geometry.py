"""Absolute align_corners=False geometry; no GT or fitting labels."""
from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn.functional as F


def identity_grid(
    height: int,
    width: int,
    *,
    batch_size: int = 1,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return an align_corners=False absolute sampling grid."""
    ys = 2.0 * (torch.arange(height, device=device, dtype=dtype) + 0.5) / height - 1.0
    xs = 2.0 * (torch.arange(width, device=device, dtype=dtype) + 0.5) / width - 1.0
    (yy, xx) = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)[None].expand(batch_size, -1, -1, -1)


def sample(field, grid):
    return F.grid_sample(
        field.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False
    )


def inside(grid, height, width):
    limit = grid.new_tensor((1 - 1 / width, 1 - 1 / height))
    return (grid.abs() <= limit + 1e-06).all(-1)


def apply_affine(matrix, grid):
    return grid @ matrix[:2, :2].T + matrix[:2, 2]


def pixel_affine_to_normalized(matrix, height, width):
    normalizer = matrix.new_tensor(
        [[2 / width, 0, 1 / width - 1], [0, 2 / height, 1 / height - 1], [0, 0, 1]]
    )
    homogeneous = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    homogeneous[:2] = matrix[:2]
    return normalizer @ homogeneous @ torch.linalg.inv(normalizer)


def calibrated_logits(logits, calibration):
    return calibration["slope"] * logits.float() + calibration["intercept"]


def affine_fields(matrix_px, height, width):
    matrix = pixel_affine_to_normalized(matrix_px.float(), height, width)
    grid = identity_grid(height, width, device=matrix.device)
    return torch.cat(
        (apply_affine(matrix, grid), apply_affine(torch.linalg.inv(matrix), grid))
    ).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class GpuLdappConfig:
    confidence_probability: float = 0.5
    target_cell_size_px: float = 35.0
    min_grid_size: int = 4
    max_grid_size: int = 16
    ransac_threshold_px: float = 3.0
    ransac_hypotheses: int = 512
    ransac_seed: int = 0
    irls_iterations: int = 3
    minimum_points: int = 12
    fdr_q: float = 0.05
    minimum_effect_px: float = 1.0
    cycle_z_threshold: float = 3.0


@dataclass
class _AffineFit:
    matrix: torch.Tensor
    inlier_ratio: torch.Tensor
    noise_components_px: torch.Tensor

    @property
    def noise_scale_px(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.noise_components_px)


def _normalized_to_pixels(grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    scale = grid.new_tensor((width, height))
    return ((grid + 1.0) * scale - 1.0) / 2.0


def _spatial_representatives(
    probability: torch.Tensor, valid: torch.Tensor, grid_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one highest-confidence point from every regular grid cell."""
    (height, width) = probability.shape
    device = probability.device
    (y, x) = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    cell_y = torch.clamp(y * grid_size // height, max=grid_size - 1)
    cell_x = torch.clamp(x * grid_size // width, max=grid_size - 1)
    groups = (cell_y * grid_size + cell_x).reshape(-1)
    scores = probability.reshape(-1).float()
    valid_flat = valid.reshape(-1)
    scores = torch.where(valid_flat, scores, scores.new_full((), -torch.inf))
    group_count = grid_size * grid_size
    maxima = scores.new_full((group_count,), -torch.inf)
    maxima.scatter_reduce_(0, groups, scores, reduce="amax", include_self=True)
    candidate = valid_flat & (scores == maxima[groups])
    flat_indices = torch.arange(scores.numel(), device=device)
    sentinel = scores.numel()
    winners = torch.where(candidate, flat_indices, sentinel)
    selected = torch.full((group_count,), sentinel, device=device, dtype=torch.long)
    selected.scatter_reduce_(0, groups, winners, reduce="amin", include_self=True)
    keep = selected < sentinel
    selected = selected[keep]
    selected_groups = torch.arange(group_count, device=device)[keep]
    points = torch.stack((selected % width, selected // width), dim=-1).float()
    return (points, probability.reshape(-1)[selected].float(), selected_groups)


def _solve_weighted_affine(
    source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, height: int, width: int
) -> torch.Tensor | None:
    center = source.new_tensor(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    coordinate_scale = float(max(height, width))
    x = torch.cat(((source - center) / coordinate_scale, torch.ones_like(source[:, :1])), dim=1)
    y = (target - center) / coordinate_scale
    weights = weights.float().clamp_min(0.0)
    if int((weights > 0.0).sum()) < 3:
        return None
    normal = x.T @ (weights[:, None] * x)
    normal = normal + torch.eye(3, device=x.device, dtype=x.dtype) * 1e-06
    right = x.T @ (weights[:, None] * y)
    try:
        beta = torch.linalg.solve(normal, right)
    except RuntimeError:
        return None
    linear = beta[:2].T
    translation = center + coordinate_scale * beta[2] - linear @ center
    matrix = torch.cat((linear, translation[:, None]), dim=1)
    return matrix if bool(torch.isfinite(matrix).all()) else None


def _apply_affine(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _fit_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    height: int,
    width: int,
    config: GpuLdappConfig,
    seed: int,
) -> _AffineFit | None:
    count = len(source)
    if count < config.minimum_points:
        return None
    device = source.device
    center = source.new_tensor(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    coordinate_scale = float(max(height, width))
    source_normalized = (source - center) / coordinate_scale
    target_normalized = (target - center) / coordinate_scale
    homogeneous = torch.cat((source_normalized, torch.ones_like(source_normalized[:, :1])), dim=1)
    generator = torch.Generator(device=device).manual_seed(seed)
    triplets = torch.randint(
        count, (config.ransac_hypotheses, 3), device=device, generator=generator
    )
    sample_x = homogeneous[triplets]
    determinant = torch.linalg.det(sample_x)
    unique = (
        (triplets[:, 0] != triplets[:, 1])
        & (triplets[:, 0] != triplets[:, 2])
        & (triplets[:, 1] != triplets[:, 2])
    )
    usable = unique & (determinant.abs() > 1e-06)
    if not bool(usable.any()):
        return None
    sample_x = sample_x[usable]
    sample_y = target_normalized[triplets[usable]]
    try:
        hypotheses = torch.linalg.solve(sample_x, sample_y)
    except RuntimeError:
        return None
    predicted = torch.matmul(homogeneous.unsqueeze(0), hypotheses)
    residual = torch.linalg.vector_norm(
        (predicted - target_normalized.unsqueeze(0)) * coordinate_scale, dim=-1
    )
    inliers = residual <= config.ransac_threshold_px
    inlier_count = inliers.sum(dim=1).float()
    truncated = residual.clamp_max(config.ransac_threshold_px).mean(dim=1)
    ranking = inlier_count - 0.1 * truncated / config.ransac_threshold_px
    best = int(ranking.argmax())
    keep = inliers[best]
    if int(keep.sum()) < config.minimum_points:
        return None
    weights = keep.float() * confidence.float().clamp_min(0.001)
    matrix = None
    for _ in range(config.irls_iterations):
        matrix = _solve_weighted_affine(source, target, weights, height, width)
        if matrix is None:
            return None
        residual_vector = _apply_affine(source, matrix) - target
        residual_norm = torch.linalg.vector_norm(residual_vector, dim=1)
        consensus_residual = residual_norm[keep]
        median = consensus_residual.median()
        mad = (consensus_residual - median).abs().median()
        robust_scale = torch.clamp(1.4826 * mad, min=0.001)
        delta = torch.clamp(3.0 * robust_scale, min=config.minimum_effect_px)
        robust_weight = torch.minimum(
            torch.ones_like(residual_norm), delta / residual_norm.clamp_min(1e-06)
        )
        weights = keep.float() * confidence.float().clamp_min(0.001) * robust_weight
    if matrix is None:
        return None
    residual_vector = _apply_affine(source, matrix) - target
    residual_norm = torch.linalg.vector_norm(residual_vector, dim=1)
    final_keep = residual_norm <= config.ransac_threshold_px
    consensus_vector = residual_vector[final_keep]
    if len(consensus_vector) < 3:
        return None
    center_vector = consensus_vector.median(dim=0).values
    component_mad = (consensus_vector - center_vector).abs().median(dim=0).values
    component_scale = torch.clamp(1.4826 * component_mad, min=0.001)
    return _AffineFit(matrix, final_keep.float().mean(), component_scale)
