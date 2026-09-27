print("Section 3 — Week 3: SAM 2 Inference Engine
Builds the SAM 2 video predictor and runs bidirectional propagation (forward + backward from the prompt slice) using the streaming memory attention across z-slices.")

%%writefile src/engine/predictor.py
import torch
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_predictor(model_cfg, ckpt_path):
    """Build a SAM 2 video predictor for bidirectional z-axis propagation.

    Specification:
    - Call build_sam2_video_predictor(model_cfg, ckpt_path, device=DEVICE).
    - model_cfg is the config filename (e.g. "configs/sam2.1/sam2.1_hiera_t.yaml"),
      resolved by SAM 2 relative to its own installed package directory via hydra
      (independent of the notebook's current working directory).
    - Return the predictor; do NOT call init_state here.

    Model size -> (cfg, ckpt) mapping (see configs/default.yaml):
        tiny  : configs/sam2.1/sam2.1_hiera_t.yaml  / checkpoints/sam2.1_hiera_tiny.pt
        small : configs/sam2.1/sam2.1_hiera_s.yaml  / checkpoints/sam2.1_hiera_small.pt
        base+ : configs/sam2.1/sam2.1_hiera_b+.yaml / checkpoints/sam2.1_hiera_base_plus.pt

    Returns:
        SAM2VideoPredictor: Predictor ready for init_state.
    """
    predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=DEVICE)
    return predictor


def build_image_predictor(model_cfg, ckpt_path):
    """Build a SAM 2 image predictor for single-frame prompted segmentation.

    Used by the LoRA training path (Week 4) instead of the video predictor,
    since SAM2ImagePredictor.set_image() alone would run under no_grad().
    """
    model = build_sam2(model_cfg, ckpt_path, device=DEVICE)
    return SAM2ImagePredictor(model)


def init_state(predictor, frames_dir):
    """Initialise the SAM 2 video memory state for a frame folder.

    Specification:
    - Call predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True,
      offload_state_to_cpu=True) -- the offload flags keep this safe on 8-16GB VRAM.
    - frames_dir must contain the 00000.png, 00001.png, ... files from
      write_frames_png (Week 2).

    Returns:
        dict: SAM 2 inference state (opaque -- pass directly to propagate).
    """
    state = predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    return state

%%writefile src/engine/propagate.py
import numpy as np
from PIL import Image


def propagate_bidirectional(predictor, state, start_z, bbox, target_hw=None):
    """Run SAM 2 bidirectional propagation and return a 3-D binary mask.

    Specification:
    Step 1 -- Inject the bounding-box prompt on start_z:
        predictor.add_new_points_or_box(state, frame_idx=start_z, obj_id=1, box=bbox)
    Step 2 -- Forward pass (start_z -> last slice), storing masks[0, 0] per frame_idx.
    Step 3 -- Backward pass (start_z -> first slice), storing masks[0, 0] per frame_idx
              (does not overwrite start_z).
    Step 4 -- Each per-slice mask is a (h, w) logit tensor on GPU:
        threshold at 0, then resize to target_hw with PIL NEAREST if it differs.
    Step 5 -- Stack all per-slice bool arrays along axis=2 -> shape (H, W, Z).

    Args:
        predictor (SAM2VideoPredictor): Built and initialised predictor.
        state (dict): Inference state from init_state.
        start_z (int): Z-index of the prompt slice.
        bbox (np.ndarray): float32 [x0, y0, x1, y1] bounding box.
        target_hw (tuple | None): (H, W) to resize masks to, or None to keep native res.

    Returns:
        np.ndarray: bool array of shape (H, W, Z) -- the full 3-D organ mask.
    """
    predictor.add_new_points_or_box(
        state, frame_idx=start_z, obj_id=1, box=bbox
    )

    results = {}

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        state, start_frame_idx=start_z, reverse=False
    ):
        results[frame_idx] = masks[0, 0]

    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        state, start_frame_idx=start_z, reverse=True
    ):
        if frame_idx == start_z:
            continue
        results[frame_idx] = masks[0, 0]

    if not results:
        raise RuntimeError("propagate_in_video returned no masks.")

    z_depth = max(results.keys()) + 1

    if target_hw is not None:
        fallback_h, fallback_w = target_hw
    else:
        sample_mask = next(iter(results.values()))
        fallback_h, fallback_w = sample_mask.shape[-2:]

    slices = []
    for z in range(z_depth):
        if z not in results:
            slices.append(np.zeros((fallback_h, fallback_w), dtype=bool))
            continue

        mask = results[z]
        mask_bool = (mask > 0).cpu().numpy()
        mask_bool = np.squeeze(mask_bool)

        if target_hw is not None and mask_bool.shape != tuple(target_hw):
            mask_u8 = (mask_bool.astype(np.uint8)) * 255
            mask_pil = Image.fromarray(mask_u8).resize(
                (target_hw[1], target_hw[0]), Image.NEAREST
            )
            mask_bool = np.array(mask_pil) > 0

        slices.append(mask_bool)

    pred3d = np.stack(slices, axis=2)
    return pred3d

%%writefile scripts/02_run_inference.py
"""02_run_inference.py -- SAM2 bidirectional propagation on one BTCV case.

Usage (run from project root):
    python scripts/02_run_inference.py --case img0001 --organ 6
    python scripts/02_run_inference.py --case img0001 --organ 6 --lora checkpoints/lora_organ6.pt

Expected output:
    - outputs/<case>/mask_organ<N>.npy    (3D bool array, shape H x W x Z)
    - outputs/<case>/overlay_organ<N>.gif (green=GT, red=prediction)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.volume_to_frames import write_frames_png
from src.data.prompts import bbox_from_mask, best_start_slice
from src.utils.viz import save_overlay_gif
from src.engine.predictor import build_predictor, init_state, DEVICE
from src.engine.propagate import propagate_bidirectional


def main():
    parser = argparse.ArgumentParser(description="SAM2 zero-shot / LoRA inference")
    parser.add_argument("--case", required=True, help="e.g. img0001")
    parser.add_argument("--organ", type=int, required=True,
                         help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--lora", default=None,
                         help="Optional: path to LoRA adapter .pt from Week 4 training")
    parser.add_argument("--img-prefix", default="img")
    parser.add_argument("--lbl-prefix", default="label")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_path = Path(cfg["paths"]["raw_images"]) / f"{args.case}.nii"
    lbl_path = Path(cfg["paths"]["raw_labels"]) / f"{args.case.replace(args.img_prefix, args.lbl_prefix)}.nii"
    frames_dir = Path(cfg["paths"]["processed"]) / args.case
    out_dir = Path("outputs") / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        print("Writing PNG frames...")
        write_frames_png(str(img_path), str(frames_dir))

    vol, _, spacing = load_volume(str(img_path))
    lbl, _, _ = load_volume(str(lbl_path))
    vol_u8 = apply_hu_window(vol)

    try:
        start_z = best_start_slice(lbl, args.organ)
    except ValueError as e:
        print(e); return

    gt_slice = (lbl[:, :, start_z] == args.organ).astype(np.uint8)
    bbox = bbox_from_mask(gt_slice, pad=4)
    if bbox is None:
        print(f"Organ {args.organ} not found in case {args.case}."); return

    print(f"Building predictor start_z={start_z} bbox={bbox}")
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])

    if args.lora:
        ckpt = torch.load(args.lora, map_location=DEVICE)
        predictor.model.image_encoder.load_state_dict(ckpt, strict=False)
        print(f"Loaded LoRA adapter from {args.lora}")

    state = init_state(predictor, str(frames_dir))

    target_hw = (vol.shape[0], vol.shape[1])
    pred3d = propagate_bidirectional(predictor, state, start_z, bbox, target_hw=target_hw)

    npy_path = out_dir / f"mask_organ{args.organ}.npy"
    np.save(str(npy_path), pred3d)
    print(f"Saved mask shape={pred3d.shape} -> {npy_path}")

    gt3d = (lbl == args.organ).astype(bool)
    gif_path = out_dir / f"overlay_organ{args.organ}.gif"
    save_overlay_gif(vol_u8, pred3d, str(gif_path), gt3d)
    print(f"Saved GIF -> {gif_path}")

    print("\nDONE -- open the GIF and check that the mask follows the organ across slices.")


if __name__ == "__main__":
    main()

!python scripts/02_run_inference.py --case {CASE} --organ {ORGAN} --img-prefix {IMG_PREFIX} --lbl-prefix {LBL_PREFIX}

from IPython.display import Image as IPyImage, display

mask_path = f"outputs/{CASE}/mask_organ{ORGAN}.npy"
gif_path = f"outputs/{CASE}/overlay_organ{ORGAN}.gif"

pred3d = np.load(mask_path)
print("Predicted mask shape:", pred3d.shape, pred3d.dtype)
print("Predicted foreground voxels:", int(pred3d.sum()))

display(IPyImage(filename=gif_path))

def week3_checklist():
    report = {
        "predictor.py (build_predictor, build_image_predictor, init_state)": True,
        "propagate.py (propagate_bidirectional)": True,
        "02_run_inference.py runs end-to-end": Path(mask_path).exists(),
        "Zero-shot mask + overlay GIF saved": Path(gif_path).exists(),
    }
    print("=" * 60)
    print("WEEK 3 COMPLETION REPORT")
    print("=" * 60)
    for item, status in report.items():
        print(f"{item:62} : {status}")
    print("=" * 60)

week3_checklist()
