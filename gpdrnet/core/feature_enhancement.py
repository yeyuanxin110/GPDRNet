# Inference-only adaptation of RoMa (Johan Edstedt, MIT).
# Modified for standalone GPDRNet; see README.md (Attribution and release license).
import math
from typing import Mapping, Sequence
import torch
from torch import nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    def __init__(self, channels: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("position embedding channels must be divisible by 4")
        self.num_pos_feats = channels // 2
        self.temperature = temperature

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        (batch_size, _, height, width) = feature.shape
        y_embed = torch.arange(1, height + 1, device=feature.device, dtype=torch.float32)[
            None, :, None
        ].expand(batch_size, -1, width)
        x_embed = torch.arange(1, width + 1, device=feature.device, dtype=torch.float32)[
            None, None, :
        ].expand(batch_size, height, -1)
        y_embed = y_embed / max(height, 1) * (2.0 * math.pi)
        x_embed = x_embed / max(width, 1) * (2.0 * math.pi)
        dim_t = torch.arange(self.num_pos_feats, device=feature.device, dtype=torch.float32)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        position = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)
        return position.to(dtype=feature.dtype)


def _partition_windows(feature: torch.Tensor, window_size: int) -> torch.Tensor:
    (batch_size, height, width, channels) = feature.shape
    feature = feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, channels
    )
    return feature.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def _merge_windows(
    windows: torch.Tensor, *, batch_size: int, height: int, width: int, window_size: int
) -> torch.Tensor:
    channels = windows.shape[-1]
    feature = windows.view(
        batch_size, height // window_size, width // window_size, window_size, window_size, channels
    )
    return feature.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)


def _pad_nhwc(feature: torch.Tensor, pad_height: int, pad_width: int) -> torch.Tensor:
    if not pad_height and (not pad_width):
        return feature
    return F.pad(feature.permute(0, 3, 1, 2), (0, pad_width, 0, pad_height)).permute(0, 2, 3, 1)


def _shifted_window_mask(
    height: int, width: int, window_size: int, shift_size: int, device: torch.device
) -> torch.Tensor:
    labels = torch.zeros((1, height, width, 1), device=device)
    height_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    width_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    label = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            labels[:, height_slice, width_slice] = label
            label += 1
    windows = _partition_windows(labels, window_size)[..., 0]
    differences = windows[:, None] - windows[:, :, None]
    return differences.masked_fill(differences != 0, -100.0)


class WindowAttentionLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        window_size: int,
        with_shift: bool,
        with_ffn: bool,
        ffn_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.with_shift = with_shift
        self.with_ffn = with_ffn
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.merge = nn.Linear(channels, channels, bias=False)
        self.norm1 = nn.LayerNorm(channels)
        if with_ffn:
            input_channels = 2 * channels
            self.mlp = nn.Sequential(
                nn.Linear(input_channels, ffn_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden_dim, channels, bias=False),
            )
            self.norm2 = nn.LayerNorm(channels)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        query = query.view(batch_size, height, width, self.channels)
        key = key.view(batch_size, height, width, self.channels)
        value = value.view(batch_size, height, width, self.channels)
        window_size = min(self.window_size, height, width)
        padded_height = math.ceil(height / window_size) * window_size
        padded_width = math.ceil(width / window_size) * window_size
        pad_height = padded_height - height
        pad_width = padded_width - width
        valid = torch.ones((batch_size, height, width, 1), device=query.device, dtype=torch.bool)
        query = _pad_nhwc(query, pad_height, pad_width)
        key = _pad_nhwc(key, pad_height, pad_width)
        value = _pad_nhwc(value, pad_height, pad_width)
        valid = _pad_nhwc(valid, pad_height, pad_width)
        shift_size = window_size // 2 if self.with_shift else 0
        if shift_size:
            shifts = (-shift_size, -shift_size)
            query = torch.roll(query, shifts=shifts, dims=(1, 2))
            key = torch.roll(key, shifts=shifts, dims=(1, 2))
            value = torch.roll(value, shifts=shifts, dims=(1, 2))
            valid = torch.roll(valid, shifts=shifts, dims=(1, 2))
        query_windows = _partition_windows(query, window_size)
        key_windows = _partition_windows(key, window_size)
        value_windows = _partition_windows(value, window_size)
        key_valid = _partition_windows(valid, window_size)[..., 0]
        scores = torch.matmul(query_windows, key_windows.transpose(1, 2)) / math.sqrt(self.channels)
        if shift_size:
            mask = _shifted_window_mask(
                padded_height, padded_width, window_size, shift_size, query.device
            )
            windows_per_image = mask.shape[0]
            scores = scores.view(
                batch_size, windows_per_image, self.window_size ** 2, self.window_size ** 2
            )
            scores = scores + mask[None]
            scores = scores.flatten(0, 1)
        scores = scores.masked_fill(~key_valid[:, None, :], -100.0)
        output = torch.matmul(scores.softmax(dim=-1), value_windows)
        output = _merge_windows(
            output,
            batch_size=batch_size,
            height=padded_height,
            width=padded_width,
            window_size=window_size,
        )
        if shift_size:
            output = torch.roll(output, shifts=(shift_size, shift_size), dims=(1, 2))
        output = output[:, :height, :width]
        return output.reshape(batch_size, height * width, self.channels)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        message = self._attention(
            self.q_proj(source),
            self.k_proj(target),
            self.v_proj(target),
            height=height,
            width=width,
        )
        message = self.norm1(self.merge(message))
        if self.with_ffn:
            message = self.norm2(self.mlp(torch.cat((source, message), dim=-1)))
        return source + message


class PairTransformerBlock(nn.Module):
    def __init__(
        self, channels: int, *, window_size: int, with_shift: bool, ffn_hidden_dim: int
    ) -> None:
        super().__init__()
        self.self_attention = WindowAttentionLayer(
            channels,
            window_size=window_size,
            with_shift=with_shift,
            with_ffn=False,
            ffn_hidden_dim=ffn_hidden_dim,
        )
        self.cross_attention_ffn = WindowAttentionLayer(
            channels,
            window_size=window_size,
            with_shift=with_shift,
            with_ffn=True,
            ffn_hidden_dim=ffn_hidden_dim,
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        source = self.self_attention(source, source, height=height, width=width)
        return self.cross_attention_ffn(source, target, height=height, width=width)


class PairFeatureTransformer(nn.Module):
    def __init__(
        self, *, channels: int, num_layers: int, window_size: int, ffn_hidden_dim: int
    ) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.layers = nn.ModuleList(
            [
                PairTransformerBlock(
                    channels,
                    window_size=window_size,
                    with_shift=layer_index % 2 == 1,
                    ffn_hidden_dim=ffn_hidden_dim,
                )
                for layer_index in range(num_layers)
            ]
        )
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self, feature_a: torch.Tensor, feature_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (batch_size, channels, height, width) = feature_a.shape
        if feature_b.shape != feature_a.shape:
            raise ValueError("paired features must have identical shapes")
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        feature_a = feature_a.flatten(2).transpose(1, 2)
        feature_b = feature_b.flatten(2).transpose(1, 2)
        source = torch.cat((feature_a, feature_b), dim=0)
        target = torch.cat((feature_b, feature_a), dim=0)
        for layer in self.layers:
            source = layer(source, target, height=height, width=width)
            target = torch.cat(source.chunk(2, dim=0)[::-1], dim=0)
        (feature_a, feature_b) = source.chunk(2, dim=0)
        feature_a = feature_a.transpose(1, 2).reshape(batch_size, channels, height, width)
        feature_b = feature_b.transpose(1, 2).reshape(batch_size, channels, height, width)
        return (feature_a.contiguous(), feature_b.contiguous())


class PairFeatureEnhancer(nn.Module):
    """Paired enhancement for a [A batch, B batch] RoMa pyramid."""

    def __init__(
        self,
        input_channels: Mapping[int, int],
        *,
        enhance_scales: Sequence[int] = (8, 4, 2),
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_heads: int = 1,
        ffn_hidden_dim: int = 512,
        window_size: int = 7,
        outer_residual_init: float = 0.01,
    ) -> None:
        super().__init__()
        if num_heads != 1:
            raise ValueError("the memory-efficient window implementation uses one head")
        self.enhance_scales = tuple((int(scale) for scale in enhance_scales))
        missing = set(self.enhance_scales) - set(input_channels)
        if missing:
            raise ValueError(f"input channel definitions missing scales {sorted(missing)}")
        self.position = PositionEmbeddingSine(hidden_dim)
        self.input_projections = nn.ModuleDict(
            {
                str(scale): nn.Conv2d(input_channels[scale], hidden_dim, 1)
                for scale in self.enhance_scales
            }
        )
        self.output_projections = nn.ModuleDict(
            {
                str(scale): nn.Conv2d(hidden_dim, input_channels[scale], 1)
                for scale in self.enhance_scales
            }
        )
        self.outer_residual = nn.ParameterDict(
            {
                str(scale): nn.Parameter(torch.tensor(outer_residual_init))
                for scale in self.enhance_scales
            }
        )
        self.transformer = PairFeatureTransformer(
            channels=hidden_dim,
            num_layers=num_layers,
            window_size=window_size,
            ffn_hidden_dim=ffn_hidden_dim,
        )

    def forward(self, pyramid: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        enhanced = dict(pyramid)
        for scale in self.enhance_scales:
            if scale not in pyramid:
                raise KeyError(f"feature pyramid is missing scale {scale}")
            pair = pyramid[scale]
            if pair.shape[0] % 2:
                raise ValueError("paired feature batch must contain an even number")
            (feature_a, feature_b) = pair.chunk(2, dim=0)
            projected_a = self.input_projections[str(scale)](feature_a)
            projected_b = self.input_projections[str(scale)](feature_b)
            position = self.position(projected_a)
            (projected_a, projected_b) = self.transformer(
                projected_a + position, projected_b + position
            )
            residual_a = self.output_projections[str(scale)](projected_a)
            residual_b = self.output_projections[str(scale)](projected_b)
            gamma = self.outer_residual[str(scale)]
            enhanced[scale] = torch.cat(
                (feature_a + gamma * residual_a, feature_b + gamma * residual_b), dim=0
            )
        return enhanced
