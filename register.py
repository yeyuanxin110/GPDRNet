#!/usr/bin/env python3
"""GPDRNet -> Learning-based LDAPP -> registered optical image and checkerboard."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gpdrnet import load_model
from gpdrnet.images import prepare_image, read_image, save_registered, save_registration_pair, warp_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sar", type=Path, required=True, help="SAR image (A)")
    parser.add_argument("--optical", type=Path, required=True, help="Optical image (B)")
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory; never overwrite"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent / "weights/registration.pth",
        help="GPDRNet state dictionary, excluding DINOv2 and LDAPP",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=Path(__file__).resolve().parent / "weights/dinov2_vitl14_pretrain.pth",
        help="Separate original DINOv2 ViT-L/14 state dictionary",
    )
    parser.add_argument(
        "--ldapp-weights",
        type=Path,
        default=Path(__file__).resolve().parent / "weights/ldapp.pth",
        help="Separate Learning-based LDAPP state dictionary",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blend", choices=("smooth", "safe", "hard"), default="smooth")
    parser.add_argument(
        "--validity", choices=("boundary", "reliable", "observed"), default="boundary",
        help="boundary: affine/dense disagreement outside predicted overlap; reliable: strict full-grid filtering; observed: legacy masks",
    )
    parser.add_argument(
        "--cycle-threshold-px", type=float, default=3.0,
        help="Maximum predicted round-trip error on the 512 grid (not an accuracy guarantee)",
    )
    parser.add_argument("--seed", type=int, default=20261002, help="RANSAC sampling seed")
    parser.add_argument(
        "--save-diagnostics", action="store_true",
        help="Also save float TIFFs, reverse warp, masks, fields and metrics; default saves only two PNGs",
    )
    parser.add_argument(
        "--optical-gray",
        action="store_true",
        help="Convert optical to grayscale, as in the OSdataset reader",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    sar = read_image(args.sar, sar=True)
    optical = read_image(args.optical, grayscale=args.optical_gray)
    images = [prepare_image(x, args.device) for x in (sar, optical)]
    model = load_model(args.weights, args.backbone, args.ldapp_weights, device=args.device)
    prediction = model(
        images[0][0], images[1][0], images[0][1], images[1][1], blend=args.blend, seed=args.seed,
        cycle_threshold_px=args.cycle_threshold_px,
    )
    # Crucially, only the LDAPP-processed field is used for these images.
    final_flow = prediction["flow"].cpu()
    mask_key = "reliable_valid" if args.validity == "reliable" else "registration_valid"
    reliable = None if args.validity == "observed" else prediction[mask_key].cpu()
    optical_on_sar, valid_a = warp_image(
        optical, sar, final_flow[0:1], registration_valid=None if reliable is None else reliable[0:1]
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save_registration_pair(args.output, optical_on_sar, valid_a, sar)
    print(f"Saved registered optical image and checkerboard to {args.output.resolve()}")
    if not args.save_diagnostics:
        return
    sar_on_optical, valid_b = warp_image(
        sar, optical, final_flow[1:2], registration_valid=None if reliable is None else reliable[1:2]
    )
    save_registered(args.output, "optical_on_sar", optical_on_sar, valid_a, sar)
    save_registered(args.output, "sar_on_optical", sar_on_optical, valid_b, optical)
    np.savez_compressed(
        args.output / "fields.npz",
        phi_sar_to_optical=final_flow[0].numpy(),
        phi_optical_to_sar=final_flow[1].numpy(),
        error_bin_probabilities_280=prediction["error_bin_probabilities"].cpu().numpy(),
        replacement_weight_512=prediction["replacement_weight"].cpu().numpy(),
        affine_sar_to_optical_normalized=prediction["affine_normalized"].cpu().numpy(),
        registration_valid_512=prediction["registration_valid"].cpu().numpy(),
        reliable_valid_512=prediction["reliable_valid"].cpu().numpy(),
        cycle_error_px_512=prediction["cycle_error_px"].cpu().numpy(),
    )
    report = {
        "model": "GPDRNet",
        "postprocessor": "Learning-based LDAPP",
        "weights": {
            "model": args.weights.name,
            "backbone": args.backbone.name,
            "ldapp": args.ldapp_weights.name,
        },
        "blend": args.blend,
        "output_validity": args.validity,
        "cycle_threshold_px_512": args.cycle_threshold_px,
        "validity_note": "No GT or flow changes. Boundary mode rejects only outside predicted affine overlap AND unreliable dense correspondence; no fit -> observed masks.",
        "ransac_seed": args.seed,
        "sar_size_hw": list(sar.valid.shape),
        "optical_size_hw": list(optical.valid.shape),
        "matching_resolution": [280, 512],
        "output_resolution": "each reference's original size",
        "flow": "absolute normalized x,y coordinates, CHW, align_corners=False; NOT displacement",
        "pixel_conversion": "x=(u+1)*moving_width/2-0.5; y=(v+1)*moving_height/2-0.5",
        "warp": {
            "optical_on_sar": "sample optical with phi_sar_to_optical",
            "sar_on_optical": "sample SAR with phi_optical_to_sar",
        },
        "confidence": "four exclusive error bins <=1, (1,3], (3,5], >5 native280 pixels; direction SAR->optical then optical->SAR",
        "fit": prediction["fit"],
        "safe_factor": prediction["safe_factor"],
        "valid_fraction": {
            "optical_on_sar": float(valid_a.mean()),
            "sar_on_optical": float(valid_b.mean()),
        },
        "seconds_including_load_io": time.perf_counter() - started,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved bidirectional registered images to {args.output.resolve()}")
    print(
        f"Blend: {args.blend}; affine fit: {prediction['fit']['reason']}; safe factor: {prediction['safe_factor']}"
    )


if __name__ == "__main__":
    main()
