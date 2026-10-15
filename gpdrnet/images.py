"""Image preparation and backward warping with absolute normalized fields."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import rasterio
import torch
import torch.nn.functional as F

from .geometry import identity_grid, inside, sample


@dataclass
class Raster:
    pixels: np.ndarray  # float32 H,W,C, original intensity (complex SAR -> amplitude)
    valid: np.ndarray
    georeference: dict


def read_image(path, *, sar=False, grayscale=False):
    path = Path(path)
    georeference = {}
    if path.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(path) as source:
            if sar or source.count == 1:
                bands = [1]
            elif source.count >= 3:
                bands = [1, 2, 3]
            else:
                raise ValueError("Optical TIFF must contain one band or at least three RGB bands")
            pixels = source.read(bands).transpose(1, 2, 0)
            valid = (source.read_masks(bands) > 0).all(0)
            if np.iscomplexobj(pixels):
                pixels = np.abs(pixels)
            if source.crs is not None:
                georeference["crs"] = source.crs
            georeference["transform"] = source.transform
        if grayscale and pixels.shape[2] == 3:
            pixels = cv2.cvtColor(pixels.astype(np.float32), cv2.COLOR_RGB2GRAY)[..., None]
    else:
        with Image.open(path) as source:
            valid = np.asarray(source.convert("RGBA"))[..., 3] > 0
            mode = "L" if sar or grayscale or source.mode in ("1", "L", "I", "F", "I;16") else "RGB"
            # Keep scalar high-bit-depth inputs; RGB conversion would quantize them.
            if source.mode in ("I", "F", "I;16"):
                pixels = np.asarray(source).copy()
            else:
                pixels = np.asarray(source.convert(mode)).copy()
            if pixels.ndim == 2:
                pixels = pixels[..., None]
    pixels = pixels.astype(np.float32)
    valid &= np.isfinite(pixels).all(2)
    if not valid.any():
        raise ValueError(f"No finite valid pixels in {path.name}")
    pixels[~valid] = 0
    return Raster(pixels, valid, georeference)


def normalize_percentile(pixels, valid):
    result = np.zeros_like(pixels, dtype=np.float32)
    for channel in range(pixels.shape[2]):
        values = pixels[..., channel]
        low, high = np.percentile(values[valid], (1.0, 99.0))
        if high > low:
            result[..., channel] = np.clip((values - low) / (high - low), 0, 1)
    result[~valid] = 0
    return result


def prepare_image(raster, device="cuda"):
    # Frozen SAR setting is intensity='auto', sar_log1p=False: no log transform.
    normalized = normalize_percentile(raster.pixels, raster.valid)
    interpolation = cv2.INTER_CUBIC if 512 > max(raster.valid.shape) else cv2.INTER_AREA
    resized = cv2.resize(normalized, (512, 512), interpolation=interpolation)
    if resized.ndim == 2:
        resized = np.repeat(resized[..., None], 3, axis=2)
    image = torch.from_numpy(resized.copy()).permute(2, 0, 1)
    mean = image.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = image.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    image = ((image - mean) / std)[None].to(device)
    valid = cv2.resize(raster.valid.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    return image, torch.from_numpy(valid.astype(bool))[None, None].to(device)


def resize_field(flow, size):
    """Resize normalized displacement, then restore the new identity grid.

    Directly resizing absolute coordinates distorts the boundary pixel centres.
    This preserves identity and constant normalized translations at every size.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("Expected absolute fields [N,2,H,W]")
    if tuple(flow.shape[-2:]) == tuple(size):
        return flow
    old_grid = identity_grid(*flow.shape[-2:], device=flow.device).permute(0, 3, 1, 2)
    new_grid = identity_grid(*size, device=flow.device).permute(0, 3, 1, 2)
    return new_grid + F.interpolate(
        flow.float() - old_grid, size, mode="bilinear", align_corners=False
    )


@torch.inference_mode()
def warp_image(moving, reference, flow, *, registration_valid=None):
    """Warp original intensities to the reference size using FINAL LDAPP flow.

    flow maps reference coordinates to positions in the moving image. This is
    backward sampling, not forward splatting and not an inverse-flow estimate.
    Pass the model's registration_valid slice to reject unreliable matches.
    Omitting it retains the legacy observed/in-bounds mask only.
    """
    flow = resize_field(flow.detach().float().cpu(), reference.valid.shape)
    if flow.shape[0] != 1:
        raise ValueError("warp_image accepts one direction at a time")
    grid = flow.permute(0, 2, 3, 1)
    image = torch.from_numpy(moving.pixels.copy()).permute(2, 0, 1)[None]
    valid_moving = torch.from_numpy(moving.valid.copy())[None, None].float()
    mask = inside(grid, *moving.valid.shape)[0]
    mask &= sample(valid_moving, grid)[0, 0] > 0.999
    mask &= torch.from_numpy(reference.valid.copy())
    if registration_valid is not None:
        if registration_valid.ndim != 4 or registration_valid.shape[:2] != (1, 1):
            raise ValueError("Expected one direction's registration_valid [1,1,H,W]")
        reliable = F.interpolate(
            registration_valid.detach().float().cpu(), reference.valid.shape,
            mode="bilinear", align_corners=False,
        )[0, 0] > 0.999
        mask &= reliable
    warped = sample(image, grid)[0].permute(1, 2, 0).numpy()
    valid = mask.numpy()
    warped[~valid] = 0
    return warped, valid


def save_registered(output, name, pixels, valid, reference):
    """Float32 TIFF preserves intensity; PNG is a percentile-stretched preview."""
    height, width, channels = pixels.shape
    path = Path(output)
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype="float32",
        compress="deflate",
        **reference.georeference,
    )
    # The validity mask, rather than intensity zero, represents nodata.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path / f"{name}.tif", "w", **profile) as target:
            target.write(pixels.transpose(2, 0, 1))
            target.write_mask(valid.astype(np.uint8) * 255)
    preview = normalize_percentile(pixels, valid) if valid.any() else np.zeros_like(pixels)
    preview = np.round(preview * 255).astype(np.uint8)
    Image.fromarray(preview[..., 0] if channels == 1 else preview).save(path / f"{name}.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(path / f"{name}_valid.png")


def save_registration_pair(output, pixels, valid, reference, *, tile_size=64):
    """Save only the registered optical PNG and its checkerboard on SAR.

    PNGs are percentile-stretched display images, not original float intensities.
    Both use the same predicted valid region; no truth or extra crop is used.
    """
    if pixels.shape[:2] != valid.shape or valid.shape != reference.valid.shape:
        raise ValueError("Registered image, validity and reference must share a grid")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")

    def rgb_preview(image, mask):
        normalized = normalize_percentile(image, mask) if mask.any() else np.zeros_like(image)
        preview = np.round(normalized * 255).astype(np.uint8)
        return np.repeat(preview, 3, axis=2) if preview.shape[-1] == 1 else preview

    optical = rgb_preview(pixels, valid)
    sar = rgb_preview(reference.pixels, reference.valid)
    yy, xx = np.indices(valid.shape)
    squares = ((xx // tile_size + yy // tile_size) % 2 == 0)[..., None]
    checkerboard = np.where(squares, sar, optical)
    checkerboard[~valid] = 0
    path = Path(output)
    Image.fromarray(optical).save(path / "optical_on_sar.png")
    Image.fromarray(checkerboard).save(path / "checkerboard_after.png")
