print("Section 4 — Week 4: LoRA Fine-Tuning
Injects LoRA adapters into the SAM 2 image encoder and fine-tunes on organ-containing slices with a combined Dice+BCE loss, bfloat16 autocast, and gradient accumulation.")

%%writefile src/train/losses.py
import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss for binary segmentation.

    Specification:
    - p = sigmoid(logits)
    - inter = (p * targets).sum(); union = p.sum() + targets.sum()
    - dice = (2*inter + eps) / (union + eps); return 1 - dice.

    Returns:
        Tensor: Scalar soft Dice loss with gradient.
    """
    p = torch.sigmoid(logits)
    inter = (p * targets).sum()
    union = p.sum() + targets.sum()
    dice_coeff = (2 * inter + eps) / (union + eps)
    return 1 - dice_coeff


def total_loss(logits, targets):
    """Combined Dice + Binary Cross-Entropy loss for organ segmentation.

    Dice drives spatial overlap; BCE drives per-pixel calibration.
    """
    return dice_loss(logits, targets) + F.binary_cross_entropy_with_logits(logits, targets)

%%writefile src/train/lora.py
import torch
import torch.nn as nn


def add_lora_peft(image_encoder, r=8, alpha=16):
    """Inject LoRA adapters into the SAM 2 image encoder via the peft library.

    Targets the fused "qkv" linear layer inside each Hiera attention block.
    After calling this, trainable_report() should show << 1% trainable params.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],
        lora_dropout=0.05,
        bias="none",
    )

    return get_peft_model(image_encoder, config)


class LoRALinear(nn.Module):
    """Wrap a fused QKV linear layer, applying low-rank updates to Q and V only.

    The Hiera attention QKV linear has out_features = 3 x in_features
    (layout [Q | K | V] along the output dimension).
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, qv_only=True):
        super().__init__()
        self.qv_only = qv_only
        self.base = base

        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scale = alpha / r
        self.dim = base.in_features

        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        delta = (x @ self.A.t()) @ self.B.t() * self.scale

        if self.qv_only:
            delta = delta.clone()
            delta[..., self.dim: 2 * self.dim] = 0.0  # zero the K slice

        return self.base(x) + delta


def inject_lora_qv(model, r=8, alpha=16):
    """Replace every attention QKV linear in the model with a LoRALinear."""
    targets = list(model.named_modules())

    for name, module in targets:
        attn = getattr(module, "attn", None)
        if attn is not None and isinstance(getattr(attn, "qkv", None), nn.Linear):
            attn.qkv = LoRALinear(attn.qkv, r=r, alpha=alpha, qv_only=True)
            continue

        if isinstance(getattr(module, "qkv", None), nn.Linear):
            module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, qv_only=True)

    return model


def trainable_report(model):
    """Print the count and fraction of trainable parameters.

    Expected: trainable well under 1% of total after LoRA injection.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0.0

    print(f"Trainable: {trainable:,} / {total:,} = {pct:.3f}%")

%%writefile src/train/train.py
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from src.train.losses import total_loss

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def _normalize(img_t):
    """Normalise (B, 3, H, W) float [0,1] tensor to ImageNet mean/std."""
    mean = _MEAN.to(img_t.device).view(1, 3, 1, 1)
    std = _STD.to(img_t.device).view(1, 3, 1, 1)
    return (img_t - mean) / std


def forward_prompted(model, img_t_norm, box_batch, image_size):
    """One SAM 2 forward pass through the internal path that keeps LoRA gradients.

    SAM2ImagePredictor.set_image() runs under torch.no_grad() -- it kills gradients.
    Use the SAM2Base internal path instead:

    1. Encode backbone -> vision_feats / feat_sizes.
    2. Reshape vision_feats from (H*W, B, C) to (B, C, H, W) per scale, coarsest = main embed.
    3. Encode the box prompt via sam_prompt_encoder.
    4. Decode to low-resolution logits via sam_mask_decoder.
    5. Upsample to (image_size, image_size) and squeeze the channel dim.

    Returns:
        Tensor: (B, H, W) logit tensor with gradient attached.
    """
    B = img_t_norm.shape[0]

    backbone_out = model.forward_image(img_t_norm)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    feats = [
        f.permute(1, 2, 0).view(B, -1, *sz)
        for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])
    ]
    feats = feats[::-1]

    image_embed = feats[-1]
    high_res_feats = feats[:-1]

    box_t = box_batch.to(img_t_norm.device).view(B, 1, 4)
    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=None, boxes=box_t, masks=None
    )

    dec_kwargs = dict(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
    )
    if high_res_feats:
        dec_kwargs["high_res_features"] = high_res_feats

    low_res_masks = model.sam_mask_decoder(**dec_kwargs)[0]

    return F.interpolate(
        low_res_masks, (image_size, image_size),
        mode="bilinear", align_corners=False
    ).squeeze(1)


def lora_state_dict(model):
    """Extract only LoRA adapter weights from the model state dict.

    Keeps keys containing ".A", ".B", "lora_A", or "lora_B" -- this is the
    file saved to checkpoints/lora_organ<N>.pt.
    """
    sd = model.image_encoder.state_dict()
    keep_markers = (".A", ".B", "lora_A", "lora_B")
    filtered = {
        k: v for k, v in sd.items()
        if any(marker in k for marker in keep_markers)
    }
    return filtered


def run_training(cfg, model, dataset, organ_id, save_path):
    """AMP + gradient-accumulation LoRA fine-tuning loop.

    1. DataLoader with cfg["train"]["micro_batch"].
    2. AdamW over LoRA params only (requires_grad == True).
    3. bfloat16 autocast forward + GradScaler-scaled backward.
    4. Optimizer step every accum_steps micro-batches, then zero_grad(set_to_none=True).
    5. Flush any leftover gradient after the last epoch.
    6. Save filtered LoRA state dict to save_path.

    Returns:
        list[float]: Mean training loss per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(
        dataset, batch_size=cfg["train"]["micro_batch"],
        shuffle=True, num_workers=0, pin_memory=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"])

    amp_enabled = cfg["train"]["amp"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    opt.zero_grad(set_to_none=True)

    accum = cfg["train"]["accum_steps"]
    global_step = 0
    epoch_losses = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for step, (img, gt, box) in enumerate(loader):
            img_t = _normalize(img.permute(0, 3, 1, 2).float().to(device))
            gt = gt.to(device, dtype=torch.float32)
            box = box.to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                logits = forward_prompted(model, img_t, box, cfg["model"]["image_size"])
                loss = total_loss(logits, gt) / accum

            scaler.scale(loss).backward()
            global_step += 1

            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum
            n_batches += 1

        mean_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        print(f"[organ {organ_id}] Epoch {epoch + 1}/{cfg['train']['epochs']} "
              f"- mean loss: {mean_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if global_step % accum != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), save_path)

    return epoch_losses

%%writefile src/eval/metrics.py
import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


def dice_score(pred3d, gt3d, eps=1e-6):
    """Volumetric Dice Similarity Coefficient between two 3-D binary masks.

    Formula: DSC = 2|A n B| / (|A| + |B|). Range 0 (no overlap) -> 1 (perfect).
    Published SAM 2 baselines: liver ~0.93, spleen ~0.94, pancreas ~0.72.
    """
    pred3d = pred3d.astype(bool)
    gt3d = gt3d.astype(bool)
    intersection = (pred3d & gt3d).sum()
    return float((2 * intersection + eps) / (pred3d.sum() + gt3d.sum() + eps))


def hd95(pred3d, gt3d, spacing):
    """95th-percentile Hausdorff Distance in millimetres via MONAI.

    Measures boundary accuracy -- lower is better.
    Typical values: liver < 5 mm, spleen < 5 mm, pancreas < 20 mm.

    MONAI expects spacing in (z, y, x) order; load_volume() returns (sx, sy, sz),
    so it is reordered before calling compute_hausdorff_distance.
    """
    p = torch.tensor(pred3d.astype(np.uint8), dtype=torch.float32)[None, None]
    g = torch.tensor(gt3d.astype(np.uint8), dtype=torch.float32)[None, None]

    sp_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

    dist = compute_hausdorff_distance(p, g, percentile=95, spacing=sp_zyx)

    return float(dist.squeeze())

print("Unit tests -- `src/eval/metrics.py`
Quick correctness check on synthetic data before trusting these metrics on real volumes.")

%%writefile tests/test_metrics.py
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import dice_score, hd95


def _cube(shape, lo, hi):
    m = np.zeros(shape, dtype=bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def test_dice_perfect_overlap():
    m = _cube((20, 20, 10), (5, 5, 2), (15, 15, 8))
    assert abs(dice_score(m, m) - 1.0) < 1e-4


def test_dice_no_overlap():
    a = _cube((20, 20, 10), (0, 0, 0), (5, 5, 5))
    b = _cube((20, 20, 10), (10, 10, 5), (15, 15, 10))
    assert dice_score(a, b) < 1e-3


def test_dice_partial_overlap_known_value():
    a = _cube((20, 20, 20), (0, 0, 0), (4, 4, 4))
    b = _cube((20, 20, 20), (2, 0, 0), (6, 4, 4))
    inter = 2 * 4 * 4  # 32
    union = 64 + 64
    expected = 2 * inter / union  # = 0.5
    assert abs(dice_score(a, b) - expected) < 1e-3


def test_dice_both_empty_is_perfect_by_convention():
    empty = np.zeros((10, 10, 5), dtype=bool)
    assert dice_score(empty, empty) > 0.99


def test_hd95_identical_masks_is_zero():
    m = _cube((20, 20, 10), (5, 5, 2), (15, 15, 8))
    d = hd95(m, m, spacing=(1.0, 1.0, 1.0))
    assert d < 1e-3


def test_hd95_shifted_masks_is_positive():
    a = _cube((30, 30, 10), (5, 5, 2), (15, 15, 8))
    b = _cube((30, 30, 10), (8, 8, 2), (18, 18, 8))  # shifted by 3 voxels
    d = hd95(a, b, spacing=(1.0, 1.0, 1.0))
    assert d > 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

!pip install -q pytest
!python -m pytest tests/test_metrics.py -v

%%writefile scripts/03_train_lora.py
"""03_train_lora.py -- fine-tune SAM2 with LoRA on BTCV organ slices.

Usage (run from project root):
    python scripts/03_train_lora.py --organ 6
    python scripts/03_train_lora.py --organ 1 --qv-only

What this script does:
1. Builds a SAM2 predictor and injects LoRA adapters into the image encoder.
2. Builds a dataset of prompted 2D organ slices from the training split.
3. Runs the AMP + gradient-accumulation training loop (bfloat16).
4. Saves the LoRA adapter weights to checkpoints/lora_organ<N>.pt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
set_seed(42)

import yaml

from src.data.dataset import BTCVSliceDataset
from src.engine.predictor import build_predictor
from src.train.lora import add_lora_peft, inject_lora_qv, trainable_report
from src.train.train import run_training


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SAM2 on BTCV")
    parser.add_argument("--organ", type=int, required=True,
                         help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--qv-only", action="store_true",
                         help="Use custom Q&V-only LoRA (Option B) instead of peft")
    parser.add_argument("--img-prefix", default="img")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    model = predictor.model

    if args.qv_only:
        inject_lora_qv(model.image_encoder,
                        r=cfg["train"]["rank"], alpha=cfg["train"]["alpha"])
    else:
        model.image_encoder = add_lora_peft(
            model.image_encoder,
            r=cfg["train"]["rank"],
            alpha=cfg["train"]["alpha"],
        )

    trainable_report(model)

    val_set = set(cfg["split"]["val_cases"])
    img_dir = Path(cfg["paths"]["raw_images"])
    lbl_dir = Path(cfg["paths"]["raw_labels"])

    all_cases = [p.stem for p in sorted(img_dir.glob("*.nii"))
                 if p.stem not in val_set]

    print(f"Training cases: {len(all_cases)} | organ={args.organ}")

    dataset = BTCVSliceDataset(
        all_cases, args.organ,
        str(img_dir), str(lbl_dir),
        image_size=cfg["model"]["image_size"],
        img_prefix=args.img_prefix,
    )
    print(f"Dataset slices: {len(dataset)}")

    if len(dataset) == 0:
        print("No slices found for this organ. Check that imagesTr/*.nii exist.")
        return

    save_path = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
    losses = run_training(cfg, model, dataset, args.organ, str(save_path))

    print(f"\nTraining complete. Final epoch loss: {losses[-1]:.4f}")
    print(f"LoRA adapter saved -> {save_path}")
    print("Run 02_run_inference.py --lora", save_path, "to test the fine-tuned model.")


if __name__ == "__main__":
    main()

!python scripts/03_train_lora.py --organ {ORGAN} --img-prefix {IMG_PREFIX}

import yaml
from src.data.nifti_io import load_volume
from src.eval.metrics import dice_score, hd95

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

VAL_CASE = cfg["split"]["val_cases"][0]
lora_ckpt = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{ORGAN}.pt"

!python scripts/02_run_inference.py --case {VAL_CASE} --organ {ORGAN} --img-prefix {IMG_PREFIX} --lbl-prefix {LBL_PREFIX}
!python scripts/02_run_inference.py --case {VAL_CASE} --organ {ORGAN} --img-prefix {IMG_PREFIX} --lbl-prefix {LBL_PREFIX} --lora {lora_ckpt}

pred_path = Path("outputs") / VAL_CASE / f"mask_organ{ORGAN}.npy"
pred3d = np.load(pred_path)

lbl_path_val = RAW_LABELS / f"{VAL_CASE.replace(IMG_PREFIX, LBL_PREFIX)}.nii"
lbl_val, _, spacing_val = load_volume(str(lbl_path_val))
gt3d_val = (lbl_val == ORGAN).astype(bool)

dsc = dice_score(pred3d, gt3d_val)
hd = hd95(pred3d, gt3d_val, spacing_val)

print(f"\nCase {VAL_CASE} | organ {ORGAN} (LoRA-finetuned, last run above)")
print(f"Dice score : {dsc:.4f}")
print(f"HD95       : {hd:.2f} mm")

def week4_checklist():
    report = {
        "losses.py (dice_loss, total_loss)": True,
        "lora.py (add_lora_peft, inject_lora_qv, trainable_report)": True,
        "train.py (forward_prompted, run_training, lora_state_dict)": True,
        "metrics.py (dice_score, hd95)": True,
        "03_train_lora.py runs end-to-end": lora_ckpt.exists(),
        "LoRA checkpoint saved": lora_ckpt.exists(),
    }
    print("=" * 60)
    print("WEEK 4 COMPLETION REPORT")
    print("=" * 60)
    for item, status in report.items():
        print(f"{item:60} : {status}")
    print("=" * 60)

week4_checklist()
