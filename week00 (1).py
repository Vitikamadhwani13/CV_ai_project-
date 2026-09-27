print("Volumetric Medical Segmentation via SAM 2 — End-to-End Pipeline (Weeks 1-6)
**BTCV Abdominal Organ Segmentation with SAM 2 streaming memory attention + LoRA fine-tuning.**

Adapted for **Kaggle Notebooks**. Before running:
1. **Settings -> Accelerator -> GPU** (T4 x2 or P100).
2. **Settings -> Internet -> On** (needed to clone SAM 2, pip install, and download the checkpoint).
3. **Add Data** -> attach a Kaggle dataset containing the BTCV `imagesTr/` + `labelsTr/` NIfTI volumes (as extracted folders, or as `imagesTr.zip` / `labelsTr.zip`), optionally with a `dataset_0.json` split file.

Run cells top to bottom. Each phase writes its own `src/...` module via `%%writefile`, then a short cell exercises it so failures are easy to localize.

| Week | Deliverable |
|---|---|
| 1 | NIfTI I/O, HU windowing, environment setup |
| 2 | Slice prompts, PNG frame export, `BTCVSliceDataset`, overlay GIFs |
| 3 | SAM 2 video predictor + bidirectional z-axis propagation |
| 4 | LoRA fine-tuning of the image encoder (Dice+BCE, AMP, grad-accum) |
| 5 | `evaluate.py`: zero-shot vs LoRA Dice / HD95 on the val split |
| 6 | Multi-organ (liver / spleen / pancreas) final report + charts |")

print("Project Overview & Reproducibility Notes

**What this notebook does:** fine-tunes SAM 2's image encoder with LoRA to segment abdominal organs
(liver, spleen, pancreas) in 3-D CT volumes from the BTCV dataset, propagating a single 2-D box prompt
through the full volume via SAM 2's bidirectional streaming memory attention. It evaluates zero-shot
vs LoRA-finetuned Dice/HD95 against the published SAM 2 baselines.

**Kaggle setup required before running (Settings panel, right sidebar):**
1. **Add Data** — attach the BTCV dataset (`imagesTr/`, `labelsTr/`, ideally `dataset_0.json` for the
   official train/val split). Phase 0 auto-detects these under `/kaggle/input` regardless of whether
   they're plain folders or zipped.
2. **Accelerator** — set to GPU (T4 x2 or P100). Phase 0.2 will warn if none is active.
3. **Internet** — must be ON (required to `pip install`, clone the SAM 2 repo, and download the
   checkpoint from Meta's servers).

**Run order:** top to bottom, once, in a single session — every `%%writefile` cell must execute before
any script that imports the file it writes. If you restart the kernel partway through, re-run from the
top (`Run All`) rather than jumping back into the middle.

**Project structure written to `/kaggle/working`:**
```
src/
  data/    nifti_io.py, prompts.py, volume_to_frames.py, dataset.py
  engine/  predictor.py, propagate.py
  train/   losses.py, lora.py, train.py
  eval/    metrics.py, evaluate.py
  utils/   seed.py, viz.py
scripts/   01_preprocess.py, 02_run_inference.py, 03_train_lora.py, 04_evaluate.py
configs/   default.yaml   (single source of truth for paths/model/train/split)
checkpoints/, outputs/    (LoRA adapters, per-case CSVs, GIFs, final report)
```

**Known limitations:**
- LoRA fine-tunes the image encoder only; SAM 2's cross-frame memory-attention module stays at its
  pretrained weights, so propagation quality is bounded by that.
- HD95 (`src/eval/metrics.py`) is computed via MONAI's `compute_hausdorff_distance` on single-channel
  binary masks — verified empirically that `include_background` has no effect in this single-channel
  case, so the default is correct as used.
- `configs/default.yaml`'s `model.cfg` path (`configs/sam2.1/sam2.1_hiera_t.yaml`) must match the
  actual layout of whatever SAM 2 checkout `pip install -e ./sam2` produces — Phase 0.4 prints the
  resolved path it found as a sanity check.
- Evaluation is wrapped in per-case try/except (Section 5) so a single corrupt case is logged and
  skipped rather than aborting an entire organ's run; check `error` column in the output CSVs for
  any skipped cases.")

print("Phase 0 — Environment & Data Setup")

import os
from pathlib import Path

PROJECT_ROOT = Path("/kaggle/working")
os.chdir(PROJECT_ROOT)

for d in ["src/data", "src/utils", "src/engine", "src/train", "src/eval",
          "scripts", "configs", "checkpoints", "outputs", "data/processed"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

for pkg in ["src", "src/data", "src/utils", "src/engine", "src/train", "src/eval"]:
    (PROJECT_ROOT / pkg / "__init__.py").touch()

print("Project root:", PROJECT_ROOT)
print("Folder scaffold ready.")

import torch

print("=" * 60)
print("GPU CHECK")
print("=" * 60)
print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name        :", torch.cuda.get_device_name(0))
    print("CUDA Version    :", torch.version.cuda)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU Memory      : {total_memory:.2f} GB")
else:
    print("WARNING: no GPU detected. Go to Settings -> Accelerator -> GPU and restart the session.")
print("=" * 60)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!pip install -q nibabel monai SimpleITK imageio einops peft hydra-core iopath tqdm pyyaml
print("Dependencies installed.")

from pathlib import Path

if not (PROJECT_ROOT / "sam2").exists():
    !git clone -q https://github.com/facebookresearch/sam2.git
!pip install -q -e ./sam2

ckpt_path = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
if not ckpt_path.exists():
    !wget -q -O {ckpt_path} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
    print("Downloaded:", ckpt_path)
else:
    print("Checkpoint already present:", ckpt_path)

candidates = list((PROJECT_ROOT / "sam2" / "sam2" / "configs").rglob("sam2.1_hiera_t.yaml"))
print("SAM2 config file(s) found:", candidates)

%%writefile src/utils/seed.py
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Configure reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    print(f"Seed set to {seed}")

from src.utils.seed import set_seed
set_seed(42)

print("Dataset auto-detection
Looks for `imagesTr/` and `labelsTr/` under `/kaggle/input` (as folders or zips), normalises `.nii.gz` to `.nii` only if needed, and resolves a train/val split (from `dataset_0.json` if present, else a standard 24/6 fallback split filtered to whichever cases actually exist).")

import shutil
import zipfile
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = PROJECT_ROOT / "data"


def find_dir(name, root):
    if not root.exists():
        return None
    if (root / name).is_dir():
        return root / name
    matches = [p for p in root.rglob(name) if p.is_dir()]
    return matches[0] if matches else None


def find_file(patterns, root):
    if not root.exists():
        return None
    for pat in patterns:
        matches = sorted(root.rglob(pat))
        if matches:
            return matches[0]
    return None


src_images_dir = find_dir("imagesTr", INPUT_ROOT)
src_labels_dir = find_dir("labelsTr", INPUT_ROOT)

if src_images_dir is None or src_labels_dir is None:
    extract_dir = DATA_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for label, patterns in [("images", ["imagesTr*.zip"]), ("labels", ["labelsTr*.zip"])]:
        zp = find_file(patterns, INPUT_ROOT)
        if zp is not None:
            print(f"Extracting {label} zip: {zp}")
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(extract_dir)

    src_images_dir = src_images_dir or find_dir("imagesTr", extract_dir)
    src_labels_dir = src_labels_dir or find_dir("labelsTr", extract_dir)

if src_images_dir is None or src_labels_dir is None:
    combo_zip = find_file(["*btcv*.zip", "*BTCV*.zip", "*Abdomen*.zip", "RawData*.zip"], INPUT_ROOT)
    if combo_zip is not None:
        extract_dir = DATA_DIR / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting combined archive: {combo_zip}")
        with zipfile.ZipFile(combo_zip) as zf:
            zf.extractall(extract_dir)
        src_images_dir = src_images_dir or find_dir("imagesTr", extract_dir)
        src_labels_dir = src_labels_dir or find_dir("labelsTr", extract_dir)

assert src_images_dir is not None, (
    "Could not locate an imagesTr/ folder under /kaggle/input -- "
    "attach the BTCV dataset to this notebook via 'Add Data'."
)
assert src_labels_dir is not None, (
    "Could not locate a labelsTr/ folder under /kaggle/input -- "
    "attach the BTCV dataset to this notebook via 'Add Data'."
)

print(f"[ok] imagesTr -> {src_images_dir}")
print(f"[ok] labelsTr -> {src_labels_dir}")

import gzip
import shutil as _shutil
from pathlib import Path


def normalize_to_nii(src_dir, dst_dir):
    plain = list(src_dir.glob("*.nii"))
    gz = list(src_dir.glob("*.nii.gz"))

    if plain and not gz:
        return src_dir  # already the right format, use in place (read-only is fine)

    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in plain:
        target = dst_dir / p.name
        if not target.exists():
            _shutil.copy2(p, target)
    for p in gz:
        target = dst_dir / p.name[:-3]  # strip ".gz"
        if not target.exists():
            with gzip.open(p, "rb") as f_in, open(target, "wb") as f_out:
                _shutil.copyfileobj(f_in, f_out)
    return dst_dir


RAW_IMAGES = normalize_to_nii(src_images_dir, DATA_DIR / "imagesTr")
RAW_LABELS = normalize_to_nii(src_labels_dir, DATA_DIR / "labelsTr")
PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

n_img = len(list(RAW_IMAGES.glob("*.nii")))
n_lbl = len(list(RAW_LABELS.glob("*.nii")))
print(f"RAW_IMAGES -> {RAW_IMAGES}  ({n_img} volumes)")
print(f"RAW_LABELS -> {RAW_LABELS}  ({n_lbl} volumes)")
assert n_img > 0 and n_lbl > 0, "No .nii volumes found after normalisation -- check the dataset contents."

import json

IMG_PREFIX, LBL_PREFIX = "img", "label"  # adjust here if your dataset uses different stems

dataset_json = find_file(["dataset*.json"], INPUT_ROOT)
existing_img_stems = {p.stem for p in RAW_IMAGES.glob("*.nii")}

if dataset_json is not None:
    split = json.load(open(dataset_json))
    train_cases = sorted(set(t["image"].split("/")[-1].split(".")[0] for t in split["training"]))
    val_cases = sorted(set(t["image"].split("/")[-1].split(".")[0] for t in split["validation"]))
    print(f"Loaded split from {dataset_json}")
else:
    train_cases = [f"{IMG_PREFIX}{n:04d}" for n in list(range(1, 11)) + list(range(21, 35))]
    val_cases = [f"{IMG_PREFIX}{n:04d}" for n in range(35, 41)]
    print("dataset_0.json not found -- using the standard BTCV 24 train / 6 val fallback split.")

train_cases = [c for c in train_cases if c in existing_img_stems]
val_cases = [c for c in val_cases if c in existing_img_stems]

if not val_cases:
    all_present = sorted(existing_img_stems)
    n_val = max(1, len(all_present) // 6)
    val_cases = all_present[-n_val:]
    train_cases = all_present[:-n_val]
    print("No split cases matched the dataset -- auto-splitting the available cases instead.")

all_cases = train_cases + val_cases
print(f"Train cases ({len(train_cases)}): {train_cases}")
print(f"Val cases   ({len(val_cases)}): {val_cases}")

CASE = val_cases[0] if val_cases else sorted(existing_img_stems)[0]
ORGAN = 6  # liver (BTCV label id); 1=spleen, 11=pancreas
print(f"\nDemo CASE = {CASE}, ORGAN = {ORGAN} (liver)")

import nibabel as nib
import numpy as np

img_path = RAW_IMAGES / f"{CASE}.nii"
lbl_path = RAW_LABELS / f"{CASE.replace(IMG_PREFIX, LBL_PREFIX)}.nii"

img = nib.as_closest_canonical(nib.load(str(img_path)))
vol = img.get_fdata().astype(np.float32)
spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
print(f"Image {img_path.name}: shape={vol.shape} spacing={spacing} HU=[{vol.min():.1f},{vol.max():.1f}]")

lbl = nib.as_closest_canonical(nib.load(str(lbl_path)))
lbl_arr = lbl.get_fdata().astype(np.int16)
print(f"Label {lbl_path.name}: shape={lbl_arr.shape} unique labels={sorted(np.unique(lbl_arr).tolist())}")

import yaml

cfg = {
    "paths": {
        "raw_images": str(RAW_IMAGES),
        "raw_labels": str(RAW_LABELS),
        "processed": str(PROCESSED),
        "checkpoints": "checkpoints",
    },
    "model": {
        "size": "tiny",  # tiny | small | base+
        "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_tiny.pt",
        "image_size": 1024,
    },
    "train": {
        "rank": 8,          # LoRA rank
        "alpha": 16,        # LoRA alpha
        "micro_batch": 1,   # per-step batch size (safe on 16GB VRAM)
        "accum_steps": 4,   # effective batch size = micro_batch * accum_steps
        "lr": 0.0001,
        "epochs": 3,
        "amp": True,        # bfloat16 autocast
    },
    "split": {
        "val_cases": val_cases,
    },
}

Path("configs/default.yaml").write_text(yaml.dump(cfg, sort_keys=False))
print("Wrote configs/default.yaml:\n")
print(Path("configs/default.yaml").read_text())

def phase0_checklist():
    report = {
        "GPU available": torch.cuda.is_available(),
        "SAM2 installed": (PROJECT_ROOT / "sam2").exists(),
        "Checkpoint downloaded": ckpt_path.exists(),
        "Image volumes found": len(list(RAW_IMAGES.glob("*.nii"))),
        "Label volumes found": len(list(RAW_LABELS.glob("*.nii"))),
        "configs/default.yaml written": Path("configs/default.yaml").exists(),
    }
    print("=" * 60)
    print("PHASE 0 COMPLETION REPORT")
    print("=" * 60)
    for item, status in report.items():
        print(f"{item:32} : {status}")
    print("=" * 60)

phase0_checklist()
