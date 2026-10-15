"""Learning-based LDAPP: affine6 correction and region/benefit gating."""
import torch
from torch import nn
import torch.nn.functional as F
from .validity import registration_validity
from .geometry import (
    identity_grid,
    sample,
    inside,
    apply_affine,
    pixel_affine_to_normalized,
    calibrated_logits,
    affine_fields,
    GpuLdappConfig,
    _fit_affine,
    _normalized_to_pixels,
    _spatial_representatives,
)

GEOMETRY_DIM = 12
TOKEN_DIM = 112
MAX_POINTS = 512
GATE_SIZE = 64
FEATURE_DIM = 25
GATE_DIM = 29
REGION_THRESHOLD = 0.5
GAIN_THRESHOLD_PX = 0.1
BLEND_WIDTH = 12


def bin_log_probabilities(logits):
    (z1, z3, z5) = logits.unbind(1)
    (above1, above3) = (F.logsigmoid(-z1), F.logsigmoid(-z3))
    return torch.stack(
        (
            F.logsigmoid(z1),
            above1 + F.logsigmoid(z3),
            above1 + above3 + F.logsigmoid(z5),
            above1 + above3 + F.logsigmoid(-z5),
        ),
        dim=1,
    )


def normalized_to_pixel_matrix(matrix, size=512):
    normalizer = matrix.new_tensor(
        [[2 / size, 0, 1 / size - 1], [0, 2 / size, 1 / size - 1], [0, 0, 1]]
    )
    return torch.linalg.inv(normalizer) @ matrix @ normalizer


def point_sample(field, grid):
    return sample(field, grid[None, :, None])[0, :, :, 0].T


@torch.no_grad()
def sparse_tokens(coarse, refined, prior_logits, matrix_px, anchors, observed, features):
    """Build tokens from predictions and visibility masks."""
    matrix = pixel_affine_to_normalized(matrix_px, 512, 512)
    (tokens, priors) = ([], [])
    (offset_y, offset_x) = torch.meshgrid(
        torch.arange(-2, 3, device=refined.device),
        torch.arange(-2, 3, device=refined.device),
        indexing="ij",
    )
    offsets = torch.stack((offset_x, offset_y), -1).reshape(25, 2).float()
    for d in range(2):
        if not len(anchors[d]):
            continue
        query = (anchors[d] + 0.5) / 256 - 1
        predicted = apply_affine(matrix if d == 0 else torch.linalg.inv(matrix), query)
        c = point_sample(coarse[d : d + 1], query)
        r = point_sample(refined[d : d + 1], query)
        q = point_sample(prior_logits[d : d + 1], query).sigmoid()
        cycle = point_sample(refined[1 - d : 2 - d], r) - query
        geometry = torch.cat(
            (
                query,
                ((c - predicted) * 32).clamp(-8, 8),
                ((r - predicted) * 32).clamp(-8, 8),
                predicted,
                q,
                (cycle * 32).clamp(-8, 8),
                q.new_full(q.shape, float(d)),
            ),
            -1,
        )
        assert geometry.shape[-1] == GEOMETRY_DIM
        parts = [geometry]
        for scale in (8, 4):
            (source, target) = features[scale]
            assert source.shape == target.shape and source.shape[0] == 2
            (height, width) = source.shape[-2:]
            assert (height, width) == (
                512 // scale,
                512 // scale,
            ), "Use FINAL512 features, never stale280"
            q_feature = F.normalize(point_sample(source[d : d + 1], query), dim=-1, eps=1e-06)
            search = predicted[:, None] + offsets[None] * predicted.new_tensor(
                (2 / width, 2 / height)
            )
            t_feature = F.normalize(
                point_sample(target[d : d + 1], search.reshape(-1, 2)), dim=-1, eps=1e-06
            )
            t_feature = t_feature.reshape(len(query), 25, -1)
            usable = inside(search, 512, 512)
            usable &= (
                point_sample(observed[1 - d : 2 - d].float(), search.reshape(-1, 2))[:, 0].reshape(
                    -1, 25
                )
                > 0.999
            )
            cost = (q_feature[:, None] * t_feature).sum(-1)
            parts.extend((torch.where(usable, cost, torch.zeros_like(cost)), usable.float()))
        tokens.append(torch.cat(parts, -1))
        priors.append(q[:, 0])
    actual = torch.cat(tokens)
    weight = torch.cat(priors)
    assert 0 < len(actual) <= MAX_POINTS and actual.shape[1] == TOKEN_DIM
    assert torch.isfinite(actual).all()
    padded = F.pad(actual, (0, 0, 0, MAX_POINTS - len(actual)))
    weights = F.pad(weight, (0, MAX_POINTS - len(actual)))
    keep = torch.arange(MAX_POINTS, device=actual.device) < len(actual)
    return (padded, weights, keep, matrix)


def observable_support(raw, candidate, observed):
    usable = observed[:, 0].clone()
    for d in range(2):
        for field in (raw, candidate):
            grid = field[d : d + 1].permute(0, 2, 3, 1)
            usable[d] &= inside(grid, 512, 512)[0]
            usable[d] &= sample(observed[1 - d : 2 - d].float(), grid)[0, 0] > 0.999
    return usable


@torch.no_grad()
def gate_features(coarse, raw, prior, candidate, observed, fit):
    grid = identity_grid(GATE_SIZE, GATE_SIZE, device=raw.device).repeat(2, 1, 1, 1)
    (c, r, a) = (sample(f, grid) for f in (coarse, raw, candidate))
    (residual_c, residual_r) = ((c - a) * 256, (r - a) * 256)
    cycle = torch.cat(
        [sample(raw[1 - d : 2 - d], r[d : d + 1].permute(0, 2, 3, 1)) for d in range(2)]
    ) - grid.permute(0, 3, 1, 2)
    cycle = cycle * 256
    dx = F.pad((residual_r[..., 1:] - residual_r[..., :-1]) / 8, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(
        (residual_r[..., 1:, :] - residual_r[..., :-1, :]) / 8, (0, 0, 0, 1), mode="replicate"
    )
    obs = [sample(observed.float(), grid)]
    for field in (r, a):
        obs.append(
            torch.cat(
                [
                    sample(observed[1 - d : 2 - d].float(), field[d : d + 1].permute(0, 2, 3, 1))
                    for d in range(2)
                ]
            )
        )
    parts = [
        grid.permute(0, 3, 1, 2),
        a,
        residual_c / 8,
        residual_r / 8,
        (c - r) * 32,
        cycle / 8,
        sample(prior, grid).sigmoid(),
        residual_c.norm(dim=1, keepdim=True) / 8,
        residual_r.norm(dim=1, keepdim=True) / 8,
        cycle.norm(dim=1, keepdim=True) / 8,
        dx,
        dy,
        *obs,
        raw.new_tensor((0.0, 1.0))[:, None, None, None].expand(2, 1, GATE_SIZE, GATE_SIZE),
        raw.new_full((2, 1, GATE_SIZE, GATE_SIZE), fit["inlier_ratio"]),
    ]
    features = torch.cat(parts, 1).clamp(-8, 8)
    assert (
        features.shape == (2, FEATURE_DIM, GATE_SIZE, GATE_SIZE) and torch.isfinite(features).all()
    )
    support = observable_support(raw, candidate, observed)
    if not fit["accepted"]:
        support.zero_()
    return (features.half().float(), support)


class RegionBenefitGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(FEATURE_DIM, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.readout = nn.Sequential(nn.Conv2d(48, 24, 1), nn.GELU(), nn.Conv2d(24, 2, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        with torch.no_grad():
            self.readout[-1].bias[0] = -4

    def forward(self, features):
        local = self.local(features)
        context = local.mean((-2, -1), keepdim=True).expand_as(local)
        output = self.readout(torch.cat((local, context), 1))
        return (output[:, :1], output[:, 1:2].tanh() * 8)


def hard_gate(region_logits, gain, support, use_gain=True):
    size = support.shape[-2:]
    probability = F.interpolate(
        region_logits.sigmoid(), size, mode="bilinear", align_corners=False
    )[:, 0]
    benefit = F.interpolate(gain, size, mode="bilinear", align_corners=False)[:, 0]
    result = support & (probability >= REGION_THRESHOLD)
    if use_gain:
        result &= benefit > GAIN_THRESHOLD_PX
    return result


def inward_blend(mask, width=BLEND_WIDTH):
    """Smooth the gate inward; this alone does not prevent folds."""
    assert width >= 1
    interior = mask[:, None].float()
    distance = torch.zeros_like(interior)
    for _ in range(width):
        interior = -F.max_pool2d(F.pad(-interior, (1, 1, 1, 1), value=0), 3, stride=1)
        distance += interior
    t = distance / width
    return (t * t * (3 - 2 * t))[:, 0]


def jacobian_determinant(flow):
    xy = flow.permute(0, 2, 3, 1) * 256
    (dx, dy) = (xy[:, :-1, 1:] - xy[:, :-1, :-1], xy[:, 1:, :-1] - xy[:, :-1, :-1])
    return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]


@torch.no_grad()
def fold_safe_blend(raw, candidate, weight, observed):
    """Backtrack to preserve positive discrete Jacobians on visible stencils.
    This is not a global nonfolding guarantee.
    """
    obs = observed[:, 0]
    stencil = obs[:, :-1, :-1] & obs[:, :-1, 1:] & obs[:, 1:, :-1]
    protected = stencil & (jacobian_determinant(raw) > 0)
    for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
        result = raw + weight[:, None] * factor * (candidate - raw)
        if not ((jacobian_determinant(result) <= 0) & protected).any():
            return (result, factor)
    return (raw.clone(), 0.0)


def multibin_distribution(logits):
    """Four exclusive EPE bins in native280 pixels."""
    assert logits.shape[1] == 3
    probability = bin_log_probabilities(logits.float()).exp()
    assert torch.isfinite(probability).all()
    return probability


@torch.no_grad()
def observable_anchors(coarse, refined, prior, observed):
    """Select anchors using source and predicted-target visibility."""
    assert coarse.shape == (2, 2, 280, 280) and refined.shape == (2, 2, 512, 512)
    grid = identity_grid(280, 280, device=coarse.device).repeat(2, 1, 1, 1)
    fine = sample(refined, grid)
    (sources, targets, weights, anchors) = ([], [], [], [])
    source_visible = F.interpolate(observed.float(), (280, 280), mode="area")[:, 0] > 0.9999
    for d in range(2):
        usable = source_visible[d].clone()
        for field in (coarse, fine):
            target = field[d].permute(1, 2, 0)
            usable &= inside(target, 512, 512)
            usable &= sample(observed[1 - d : 2 - d].float(), target[None])[0, 0] > 0.999
        (points, scores, _) = _spatial_representatives(prior[d, 0], usable, 16)
        (x, y) = (points[:, 0].long(), points[:, 1].long())
        physical = (points + 0.5) * (512 / 280) - 0.5
        target = _normalized_to_pixels(coarse[d, :, y, x].T, 512, 512)
        sources.append(physical if d == 0 else target)
        targets.append(target if d == 0 else physical)
        weights.append(scores.sigmoid().clamp_min(0.001))
        anchors.append(physical)
    return (torch.cat(sources), torch.cat(targets), torch.cat(weights), anchors)


def fit_affine6(source, target, confidence, seed):
    config = GpuLdappConfig(
        minimum_points=24, ransac_threshold_px=3.0, ransac_hypotheses=512, irls_iterations=3
    )
    fit = _fit_affine(source, target, confidence, 512, 512, config, seed)
    info = {
        "accepted": False,
        "points": len(source),
        "inlier_ratio": 0.0,
        "reason": "failed_fit",
        "family": "affine6",
    }
    if fit is None:
        return (None, info)
    singular = torch.linalg.svdvals(fit.matrix[:, :2])
    if (
        not torch.isfinite(fit.matrix).all()
        or torch.linalg.det(fit.matrix[:, :2]) <= 0
        or singular.min() < 0.25
        or (singular.max() > 4)
    ):
        return (None, {**info, "reason": "invalid_geometry"})
    matrix = torch.eye(3, device=source.device)
    matrix[:2] = fit.matrix
    return (
        matrix,
        {
            **info,
            "accepted": True,
            "reason": "accepted",
            "inlier_ratio": float(fit.inlier_ratio),
            "initial_pixels": matrix.cpu().tolist(),
        },
    )


@torch.no_grad()
def multibin_tokens(coarse, raw, prior, bins, matrix, anchors, observed, features):
    (old, weights, keep, initial) = sparse_tokens(
        coarse, raw, prior, matrix, anchors, observed, features
    )
    parts = [
        point_sample(bins[d : d + 1], (points + 0.5) / 256 - 1)
        for (d, points) in enumerate(anchors)
        if len(points)
    ]
    extra = torch.cat(parts)
    extra = F.pad(extra, (0, 0, 0, MAX_POINTS - len(extra)))
    return (torch.cat((old, extra), -1), weights, keep, initial)


class MultibinAffine6(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(116, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.readout = nn.Sequential(nn.Linear(134, 64), nn.GELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, tokens, weights, keep, initial):
        assert tokens.shape[-1] == 116 and keep.any(1).all()
        hidden = self.point_encoder(tokens)
        weight = weights.clamp_min(0.05) * keep.float()
        weight = weight / weight.sum(1, keepdim=True)
        mean = (hidden * weight[..., None]).sum(1)
        variance = ((hidden - mean[:, None]).square() * weight[..., None]).sum(1)
        parameters = self.readout(
            torch.cat((mean, (variance + 1e-06).sqrt(), initial[:, :2].flatten(1)), -1)
        ).tanh()
        linear = torch.matrix_exp(parameters[:, :4].reshape(-1, 2, 2) * 0.02)
        translation = parameters[:, 4:] * (8 / 256)
        upper = torch.cat((linear, translation[..., None]), -1)
        bottom = initial.new_tensor((0.0, 0.0, 1.0)).expand(len(initial), 1, 3)
        return (torch.cat((upper, bottom), 1) @ initial, parameters)


class MultibinRegionGate(RegionBenefitGate):
    def __init__(self):
        super().__init__()
        self.local[0] = nn.Conv2d(GATE_DIM, 24, 3, padding=1)


@torch.no_grad()
def corrected_inputs(data, corrector):
    (raw, fit) = (data["raw"], data["fit"])
    initial = candidate = raw.clone()
    matrix = data["initial"]
    if fit["accepted"]:
        (matrix, _) = corrector(
            *(data[key][None] for key in ("tokens", "weights", "keep", "initial"))
        )
        matrix = matrix[0]
        initial = affine_fields(normalized_to_pixel_matrix(data["initial"]), 512, 512)
        candidate = affine_fields(normalized_to_pixel_matrix(matrix), 512, 512)
    (old, support) = gate_features(
        data["coarse"], raw, data["prior"], candidate, data["observed"], fit
    )
    grid = identity_grid(GATE_SIZE, GATE_SIZE, device=raw.device).repeat(2, 1, 1, 1)
    features = torch.cat((old, sample(data["bins"], grid)), 1).half().float()
    assert features.shape[1] == GATE_DIM
    support &= observable_support(raw, initial, data["observed"])
    return (initial, candidate, features, support, matrix)


class LearningBasedLDAPP(nn.Module):
    """RANSAC initialization -> learned affine6 -> learned gating."""

    def __init__(self):
        super().__init__()
        self.corrector = MultibinAffine6()
        self.gate = MultibinRegionGate()
        self.register_buffer("calibration", torch.zeros(2, dtype=torch.float64))

    @torch.no_grad()
    def forward(self, coarse, raw, logits, observed, features, *, blend="smooth", seed=20261002,
                cycle_threshold_px=3.0):
        if blend not in ("smooth", "safe", "hard"):
            raise ValueError("Unknown blend mode")
        prior = calibrated_logits(
            logits[:, :1],
            {"slope": float(self.calibration[0]), "intercept": float(self.calibration[1])},
        )
        bins = multibin_distribution(logits)
        source, target, weights, anchors = observable_anchors(coarse, raw, prior, observed)
        matrix, fit = fit_affine6(source, target, weights, int(seed))
        if matrix is None:
            tokens = raw.new_zeros(MAX_POINTS, 116)
            weights = raw.new_zeros(MAX_POINTS)
            keep = torch.zeros(MAX_POINTS, dtype=torch.bool, device=raw.device)
            keep[0] = True
            initial = torch.eye(3, device=raw.device)
        else:
            tokens, weights, keep, initial = multibin_tokens(
                coarse, raw, prior, bins, matrix, anchors, observed, features
            )
        data = dict(
            coarse=coarse,
            raw=raw,
            prior=prior,
            bins=bins,
            observed=observed,
            tokens=tokens,
            weights=weights,
            keep=keep,
            initial=initial,
            fit=fit,
        )
        _, candidate, gate_input, support, matrix = corrected_inputs(data, self.corrector)
        region_logits, gain = self.gate(gate_input)
        accepted = hard_gate(region_logits, gain, support)
        weight = inward_blend(accepted)
        factor = 1.0
        if blend == "hard":
            flow = torch.where(accepted[:, None], candidate, raw)
            weight = accepted.float()
        elif blend == "safe":
            flow, factor = fold_safe_blend(raw, candidate, weight, observed)
            weight = weight * factor
        else:
            flow = raw + weight[:, None] * (candidate - raw)
        if not torch.isfinite(flow).all():
            raise FloatingPointError("Nonfinite Learning-based LDAPP output")
        validity = registration_validity(
            flow, observed, affine_flow=candidate if fit["accepted"] else None,
            cycle_threshold_px=cycle_threshold_px,
        )
        return dict(
            flow=flow,
            registration_valid=validity["boundary_valid"],
            reliable_valid=validity["valid"],
            cycle_error_px=validity["cycle_error_px"],
            error_bin_probabilities=bins,
            replacement_weight=weight,
            affine_normalized=matrix,
            fit=fit,
            safe_factor=factor,
        )
