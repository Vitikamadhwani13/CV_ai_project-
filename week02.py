print("Section 2 — Week 2: Preprocessing Pipeline
Bounding-box prompts from masks, volume-to-PNG-frame export, the `BTCVSliceDataset` used for LoRA training, and the `01_preprocess.py` CLI.")

%%writefile src/data/prompts.py
import numpy as np
from scipy import ndimage


def bbox_from_mask(mask2d, pad=4):
    """Return a SAM 2-format bounding box around the foreground of a 2-D mask.

    Specification:
    - Find all True/nonzero pixels; return None if there are none.
    - Compute the tight axis-aligned bounding box (min/max row and column).
    - Expand every side by pad pixels, clamped to image bounds.
    - Return format: float32 array [x0, y0, x1, y1] (x = column, y = row).
    """
    mask2d = np.asarray(mask2d)

    if mask2d.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask2d.shape}")

    ys, xs = np.nonzero(mask2d)

    if ys.size == 0:
        return None

    h, w = mask2d.shape

    y0_tight, y1_tight = int(ys.min()), int(ys.max())
    x0_tight, x1_tight = int(xs.min()), int(xs.max())

    x0 = max(0, x0_tight - pad)
    y0 = max(0, y0_tight - pad)
    x1 = min(w, x1_tight + 1 + pad)
    y1 = min(h, y1_tight + 1 + pad)

    return np.array([x0, y0, x1, y1], dtype=np.float32)


def best_start_slice(label_vol, organ_id):
    """Return the axial index of the slice with the largest organ cross-section.

    Specification:
    - Binarise label_vol to the target organ (label_vol == organ_id).
    - Raise ValueError if the organ is absent.
    - For each z-slice compute the foreground pixel count.
    - Return the z index with the maximum count (argmax across the Z axis).
    """
    label_vol = np.asarray(label_vol)
    organ_mask = (label_vol == organ_id)

    if not np.any(organ_mask):
        raise ValueError(f"Organ id {organ_id} not present in label volume.")

    labeled, n_components = ndimage.label(organ_mask)
    if n_components == 0:
        raise ValueError(f"Organ id {organ_id} not present in label volume.")

    per_slice_counts = organ_mask.sum(axis=(0, 1))

    return int(np.argmax(per_slice_counts))

%%writefile src/data/volume_to_frames.py
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb


def write_frames_png(nifti_path, out_dir):
    """Convert every axial slice of a CT volume to a numbered PNG file on disk.

    Specification:
    - Load and HU-window the volume using Week 1 utilities.
    - Convert each (H, W) greyscale slice to (H, W, 3) RGB via to_rgb().
    - Save as PNG named 00000.png, 00001.png, ... (5-digit zero-padded index = z).
    - Create out_dir if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol, affine, spacing = load_volume(str(nifti_path))
    vol_u8 = apply_hu_window(vol)

    z_depth = vol_u8.shape[2]

    for z in range(z_depth):
        slice_u8 = vol_u8[:, :, z]
        rgb = to_rgb(slice_u8)

        frame_path = out_dir / f"{z:05d}.png"
        Image.fromarray(rgb).save(frame_path)

    return z_depth


def frames_to_arrays(frames_dir):
    """Load all PNG frames from a folder into a single uint8 numpy array.

    Specification:
    - Read every .png in frames_dir in sorted (filename) order.
    - Stack along a new leading axis -> shape (Z, H, W, 3).
    """
    frames_dir = Path(frames_dir)

    png_files = sorted(frames_dir.glob("*.png"))

    if not png_files:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    frames = []
    for f in png_files:
        img = Image.open(f).convert("RGB")
        frames.append(np.array(img, dtype=np.uint8))

    return np.stack(frames, axis=0)

%%writefile src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.prompts import bbox_from_mask


class BTCVSliceDataset(Dataset):
    """2-D prompted dataset of organ-containing axial slices for LoRA training.

    __init__:
    - Iterate over every case in `cases`.
    - Load image and label volumes from image_dir/<case>.nii and
      label_dir/<case-with-img-replaced-by-label>.nii.
    - For each z-slice where organ_id is present:
        * HU-window and convert the image slice to uint8 RGB (H, W, 3).
        * Resize image to (image_size, image_size) using PIL BILINEAR.
        * Extract the binary GT mask; resize with PIL NEAREST to preserve labels.
        * Compute bbox_from_mask on the resized GT; skip if None.
        * Store (resized_img_array, resized_gt_array, bbox_array) in self.samples.

    __getitem__(idx) returns (img_tensor, gt_tensor, box_tensor):
        img_tensor : torch.float32, shape (image_size, image_size, 3), range [0, 1]
        gt_tensor  : torch.float32, shape (image_size, image_size), binary 0/1
        box_tensor : torch.float32, shape (4,), [x0, y0, x1, y1]
    """

    def __init__(self, cases, organ_id, image_dir, label_dir, image_size=1024,
                 img_prefix="img", lbl_prefix="label"):
        self.samples = []
        self.organ_id = organ_id
        self.image_size = image_size

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        for case in cases:
            img_path = image_dir / f"{case}.nii"
            label_stem = case.replace(img_prefix, lbl_prefix)
            lbl_path = label_dir / f"{label_stem}.nii"

            if not img_path.exists():
                print(f"  [skip] image not found: {img_path}")
                continue
            if not lbl_path.exists():
                print(f"  [skip] label not found: {lbl_path}")
                continue

            print(f"  Loading case {case} ...")

            vol, _, _ = load_volume(str(img_path))
            lbl_vol, _, _ = load_volume(str(lbl_path))

            vol_u8 = apply_hu_window(vol)
            lbl_vol = lbl_vol.astype(np.int32)

            z_depth = vol_u8.shape[2]
            n_added = 0

            for z in range(z_depth):
                gt_slice = (lbl_vol[:, :, z] == organ_id)

                if not np.any(gt_slice):
                    continue

                slice_u8 = vol_u8[:, :, z]
                rgb = np.repeat(slice_u8[..., None], 3, axis=2).astype(np.uint8)
                img_pil = Image.fromarray(rgb).resize(
                    (image_size, image_size), Image.BILINEAR
                )
                img_resized = np.array(img_pil, dtype=np.uint8)

                gt_u8 = (gt_slice.astype(np.uint8)) * 255
                gt_pil = Image.fromarray(gt_u8).resize(
                    (image_size, image_size), Image.NEAREST
                )
                gt_resized = (np.array(gt_pil, dtype=np.uint8) > 0).astype(np.uint8)

                bbox = bbox_from_mask(gt_resized)
                if bbox is None:
                    continue

                self.samples.append((img_resized, gt_resized, bbox))
                n_added += 1

            print(f"    -> {n_added} organ-containing slices added")

        print(f"BTCVSliceDataset: {len(self.samples)} total samples "
              f"across {len(cases)} case(s), organ_id={organ_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_arr, gt_arr, bbox = self.samples[idx]

        img_tensor = torch.from_numpy(img_arr.astype(np.float32) / 255.0)
        gt_tensor = torch.from_numpy(gt_arr.astype(np.float32))
        box_tensor = torch.from_numpy(bbox.astype(np.float32))

        return img_tensor, gt_tensor, box_tensor

%%writefile scripts/01_preprocess.py
"""01_preprocess.py -- preprocess a single BTCV case to PNG frames + overlay.

Usage (run from project root):
    python scripts/01_preprocess.py --case img0001 --organ 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import imageio.v2 as imageio

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb
from src.data.volume_to_frames import write_frames_png
from src.data.prompts import bbox_from_mask, best_start_slice


def draw_rect(img, x0, y0, x1, y1, color=(0, 255, 0), thickness=2):
    """Draw a filled-border rectangle on an (H, W, 3) uint8 array in-place."""
    img[y0:y0 + thickness, x0:x1] = color
    img[y1 - thickness:y1, x0:x1] = color
    img[y0:y1, x0:x0 + thickness] = color
    img[y0:y1, x1 - thickness:x1] = color


def main():
    parser = argparse.ArgumentParser(description="Preprocess one BTCV case")
    parser.add_argument("--case", required=True, help="e.g. img0001")
    parser.add_argument("--organ", type=int, default=6,
                         help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--img-prefix", default="img")
    parser.add_argument("--lbl-prefix", default="label")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_path = Path(cfg["paths"]["raw_images"]) / f"{args.case}.nii"
    lbl_path = Path(cfg["paths"]["raw_labels"]) / f"{args.case.replace(args.img_prefix, args.lbl_prefix)}.nii"
    out_dir = Path(cfg["paths"]["processed"]) / args.case

    print(f"Loading {img_path}")
    vol, affine, spacing = load_volume(str(img_path))
    print(f"  shape={vol.shape} spacing={tuple(f'{s:.2f}' for s in spacing)} mm")

    n = write_frames_png(str(img_path), str(out_dir))
    print(f"  Wrote {n} PNG frames -> {out_dir}")

    if not lbl_path.exists():
        print("  No label file found -- skipping overlay.")
        return

    lbl, _, _ = load_volume(str(lbl_path))
    try:
        start_z = best_start_slice(lbl, args.organ)
    except ValueError as e:
        print(f"  {e}")
        return

    vol_u8 = apply_hu_window(vol)
    slice_u8 = vol_u8[:, :, start_z]
    rgb = to_rgb(slice_u8).copy()

    gt_slice = (lbl[:, :, start_z] == args.organ).astype(np.uint8)
    rgb[gt_slice == 1] = (
        rgb[gt_slice == 1] * 0.5 + np.array([220, 60, 60]) * 0.5
    ).astype(np.uint8)

    bbox = bbox_from_mask(gt_slice, pad=4)
    if bbox is not None:
        x0, y0, x1, y1 = bbox.astype(int)
        draw_rect(rgb, x0, y0, x1, y1, color=(0, 255, 0), thickness=2)

    overlay_path = out_dir / "overlay.png"
    imageio.imwrite(str(overlay_path), rgb)
    print(f"  Best start slice: z={start_z} bbox={bbox}")
    print(f"  Saved overlay -> {overlay_path}")
    print("\nDONE -- open overlay.png and check that the green box surrounds the organ.")


if __name__ == "__main__":
    main()

import importlib
import src.data.prompts, src.data.volume_to_frames, src.data.dataset

importlib.reload(src.data.prompts)
importlib.reload(src.data.volume_to_frames)
importlib.reload(src.data.dataset)

from src.data.prompts import bbox_from_mask, best_start_slice
from src.data.volume_to_frames import write_frames_png, frames_to_arrays
from src.data.dataset import BTCVSliceDataset

toy_mask = np.zeros((20, 20), dtype=np.uint8)
toy_mask[5:10, 8:14] = 1
box = bbox_from_mask(toy_mask, pad=2)
assert box is not None
assert bbox_from_mask(np.zeros((10, 10))) is None
print("bbox_from_mask sanity check passed:", box)
print("Week 2 modules import cleanly.")

!python scripts/01_preprocess.py --case {CASE} --organ {ORGAN} --img-prefix {IMG_PREFIX} --lbl-prefix {LBL_PREFIX}

import matplotlib.pyplot as plt
import imageio.v2 as imageio
overlay = imageio.imread(f"{PROCESSED}/{CASE}/overlay.png")
plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title(f"{CASE} — prompt slice + bbox")
plt.axis("off")
plt.show()

frames_arr = frames_to_arrays(f"{PROCESSED}/{CASE}")
print("frames_to_arrays shape:", frames_arr.shape, frames_arr.dtype)

dataset = BTCVSliceDataset(
    cases=[CASE],
    organ_id=ORGAN,
    image_dir=str(RAW_IMAGES),
    label_dir=str(RAW_LABELS),
    image_size=1024,
    img_prefix=IMG_PREFIX,
    lbl_prefix=LBL_PREFIX,
)
print("Dataset length:", len(dataset))

img_t, gt_t, box_t = dataset[0]
print("img_tensor :", img_t.shape, img_t.dtype, img_t.min().item(), img_t.max().item())
print("gt_tensor  :", gt_t.shape, gt_t.dtype, gt_t.unique())
print("box_tensor :", box_t.shape, box_t)

def week2_checklist():
    report = {
        "prompts.py (bbox_from_mask, best_start_slice)": True,
        "volume_to_frames.py (write_frames_png, frames_to_arrays)": True,
        "dataset.py (BTCVSliceDataset)": True,
        "01_preprocess.py runs end-to-end": Path(f"{PROCESSED}/{CASE}/overlay.png").exists(),
    }
    print("=" * 60)
    print("WEEK 2 COMPLETION REPORT")
    print("=" * 60)
    for item, status in report.items():
        print(f"{item:52} : {status}")
    print("=" * 60)

week2_checklist()
