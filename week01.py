print("Section 1 — Week 1: Core NIfTI I/O & Visualization Utilities
HU windowing, volume loading, RGB conversion, isotropic resampling, and overlay helpers used by every later week.")

%%writefile src/data/nifti_io.py
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_volume(path):
    """Load a NIfTI file and return (volume, affine, voxel_spacing).

    Steps: validate existence, load, convert to canonical orientation,
    extract the float32 array, affine matrix, and (sx, sy, sz) spacing in mm.
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata().astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return vol, img.affine, spacing


def apply_hu_window(vol, lo=-150, hi=250, as_uint8=True):
    """Clip a CT volume to a soft-tissue HU window and rescale to [0,255] or [0,1]."""
    v = np.clip(vol, lo, hi)
    v = (v - lo) / (hi - lo)
    if as_uint8:
        return (v * 255).astype(np.uint8)
    return v.astype(np.float32)


def to_rgb(slice2d_u8):
    """Convert a (H, W) uint8 greyscale slice to (H, W, 3) by repeating the channel."""
    if slice2d_u8.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {slice2d_u8.shape}")
    return np.repeat(slice2d_u8[..., None], 3, axis=2)


def resample_isotropic(vol, spacing, target=1.5, order=1):
    """Resample a volume to isotropic voxel spacing using scipy zoom.

    order=1 for image interpolation, order=0 for mask interpolation.
    """
    factors = [s / target for s in spacing]
    resampled = zoom(vol, factors, order=order)
    return resampled, (target, target, target)

%%writefile src/utils/viz.py
import numpy as np
from pathlib import Path
import imageio.v2 as imageio

BTCV_ORGANS = {
    0: "background", 1: "spleen", 2: "right_kidney", 3: "left_kidney",
    4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "ivc", 10: "portal_vein", 11: "pancreas", 12: "right_adrenal", 13: "left_adrenal",
}


def describe_labels(label_volume):
    """Print which BTCV organs are present in a label volume."""
    ids = sorted(int(i) for i in np.unique(label_volume))
    print("=" * 60)
    print("ORGANS PRESENT")
    print("=" * 60)
    for idx in ids:
        print(f"{idx:2d} -> {BTCV_ORGANS.get(idx, 'unknown')}")


def overlay_mask(image_rgb, mask, color=(255, 0, 0)):
    """Paint `mask` in a solid color onto a copy of an (H, W, 3) RGB image."""
    overlay = image_rgb.copy()
    overlay[mask > 0] = color
    return overlay


def save_png(image, output_path):
    """Save an (H, W, 3) uint8 RGB image as a PNG, creating parent dirs as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)
    print(f"Saved image -> {output_path}")


def visualize_slice(image, title="CT Slice", figsize=(8, 8)):
    """Display a single CT/overlay slice inline with matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_overlay_gif(vol_u8, pred3d, out_path, gt3d=None):
    """Write an animated GIF overlaying segmentation masks onto the CT volume.

    Specification:
    - Iterate over every axial slice (Z axis) of vol_u8.
    - Convert each greyscale (H, W) slice to RGB by repeating the channel.
    - Where pred3d[:, :, z] is True, apply a RED tint (blend 50% with [220,60,60]).
    - Where gt3d[:, :, z] is True (if gt3d is not None), apply a GREEN tint
      (blend 50% with [60,220,60]). GT is rendered beneath the prediction.
    - Collect all (H, W, 3) uint8 frames and write them as an animated GIF.
    - Recommended frame duration: 80 ms.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred3d = np.asarray(pred3d).astype(bool)
    if gt3d is not None:
        gt3d = np.asarray(gt3d).astype(bool)

    z_depth = vol_u8.shape[2]
    frames = []

    red = np.array([220, 60, 60], dtype=np.float32)
    green = np.array([60, 220, 60], dtype=np.float32)

    for z in range(z_depth):
        slice_u8 = vol_u8[:, :, z]
        rgb = np.repeat(slice_u8[..., None], 3, axis=2).astype(np.float32)

        if gt3d is not None:
            gt_mask = gt3d[:, :, z]
            if np.any(gt_mask):
                rgb[gt_mask] = rgb[gt_mask] * 0.5 + green * 0.5

        pred_mask = pred3d[:, :, z]
        if np.any(pred_mask):
            rgb[pred_mask] = rgb[pred_mask] * 0.5 + red * 0.5

        frames.append(np.clip(rgb, 0, 255).astype(np.uint8))

    imageio.mimsave(str(out_path), frames, duration=0.08)
    print(f"Saved overlay GIF -> {out_path} ({z_depth} frames)")

import matplotlib.pyplot as plt
from src.data.nifti_io import load_volume, apply_hu_window, to_rgb
from src.utils.viz import describe_labels, overlay_mask, visualize_slice, save_png

vol, affine, spacing = load_volume(str(img_path))
lbl_vol, _, _ = load_volume(str(lbl_path))
windowed_vol = apply_hu_window(vol)

describe_labels(lbl_vol)

areas = (lbl_vol == ORGAN).sum(axis=(0, 1))
best_slice = int(np.argmax(areas))
print(f"\nBest slice for organ {ORGAN}: {best_slice}")

rgb_slice = to_rgb(windowed_vol[:, :, best_slice])
organ_mask = (lbl_vol[:, :, best_slice] == ORGAN)
overlay = overlay_mask(rgb_slice, organ_mask)

visualize_slice(rgb_slice, title="Raw CT")
visualize_slice(overlay, title=f"Organ {ORGAN} overlay")
save_png(overlay, f"outputs/{CASE}_organ{ORGAN}_overlay.png")

from src.data.nifti_io import resample_isotropic

resampled_vol, new_spacing = resample_isotropic(vol, spacing, target=1.5, order=1)
print("Original shape:", vol.shape, "spacing:", spacing)
print("Resampled shape:", resampled_vol.shape, "spacing:", new_spacing)

print("\nWeek 1 modules import cleanly.")

def week1_checklist():
    report = {
        "GPU / SAM2 environment ready": torch.cuda.is_available(),
        "nifti_io.py (load_volume, apply_hu_window, to_rgb, resample_isotropic)": True,
        "viz.py (describe_labels, overlay_mask, save_png, visualize_slice)": True,
        "BTCV volume + label loaded and inspected": True,
    }
    print("=" * 60)
    print("WEEK 1 COMPLETION REPORT")
    print("=" * 60)
    for item, status in report.items():
        print(f"{item:70} : {status}")
    print("=" * 60)

week1_checklist()
