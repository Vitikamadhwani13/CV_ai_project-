print("Section 5 — Weeks 5 & 6: Evaluation Harness & Multi-Organ Final Report
Week 5 built the core `evaluate.py` (zero-shot vs LoRA Dice/HD95 on the val split). Week 6 upgraded the CLI driver with CSV output, LoRA auto-detection, and published-baseline printing, then ran it across liver, spleen and pancreas for the final report below. Only the final version of each file is shipped here.

Note on the propagation path: LoRA fine-tuning in Week 4 targeted the image encoder only — the SAM 2 memory-attention module keeps its pretrained, zero-shot weights. So `evaluate_organ` below runs the exact same bidirectional, cross-frame memory-attention propagation for both the zero-shot and LoRA branches; only the per-frame image features going into that propagation differ.")

%%writefile src/eval/evaluate.py
import gc
import numpy as np
from pathlib import Path

import torch

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.prompts import bbox_from_mask, best_start_slice
from src.data.volume_to_frames import write_frames_png
from src.engine.predictor import build_predictor, init_state, DEVICE
from src.engine.propagate import propagate_bidirectional
from src.eval.metrics import dice_score, hd95
from src.utils.viz import save_overlay_gif


def _free_state(predictor, state):
    """Release a SAM2 video inference state and reclaim GPU memory.

    Called after every propagate_bidirectional() call in the eval loop --
    each state holds cached per-frame features, and across a multi-organ,
    multi-case run these add up fast enough to risk OOM on a 16GB GPU if
    left to Python's garbage collector alone.
    """
    try:
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
    except Exception:
        pass
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_predictor(cfg, lora_path=None):
    """Build a SAM 2 video predictor and optionally load a LoRA adapter.

    Specification:
    - predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    - If lora_path is not None and the file exists:
        adapter = torch.load(lora_path, map_location=DEVICE)
        predictor.model.image_encoder.load_state_dict(adapter, strict=False)
    - Return predictor.

    Returns:
        SAM2VideoPredictor (with LoRA weights loaded if lora_path given).
    """
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])

    if lora_path is not None and Path(lora_path).exists():
        adapter = torch.load(lora_path, map_location=DEVICE)
        predictor.model.image_encoder.load_state_dict(adapter, strict=False)
        print(f"  Loaded LoRA adapter -> {lora_path}")

    return predictor


def evaluate_organ(cfg, organ_id, lora_path=None, img_prefix="img", lbl_prefix="label"):
    """Run zero-shot and optional LoRA inference on all validation cases.

    Specification:
    - val_cases = cfg["split"]["val_cases"]
    - pred_zs = _load_predictor(cfg); pred_lora = _load_predictor(cfg, lora_path) if lora_path.
    For each case in val_cases:
        a. Resolve paths; skip if either image or label file does not exist.
        b. Write PNG frames if not already done.
        c. Load volume + labels, HU-window, build gt3d = (lbl == organ_id); skip if empty.
        d. Find prompt: start_z = best_start_slice(...), bbox = bbox_from_mask(...); skip if None.
        e. Zero-shot: propagate_bidirectional with pred_zs -> dsc_zs, hd_zs; save GIF.
        f. LoRA (if pred_lora given): propagate_bidirectional with pred_lora -> dsc_lo, hd_lo;
           save GIF. Otherwise dsc_lo = hd_lo = NaN.
        g. Append dict(case, dsc_zs, hd95_zs, dsc_lora, hd95_lora) to results.
    A single case's failure (corrupt volume, OOM, etc.) is caught and logged
    so it doesn't abort the whole organ's evaluation -- the case is recorded
    with NaN metrics and a note, and the run continues.

    Returns:
        list[dict]: One dict per processed val case with keys
            case, dsc_zs, hd95_zs, dsc_lora, hd95_lora, error (None on success).
    """
    val_cases = cfg["split"]["val_cases"]
    img_dir = Path(cfg["paths"]["raw_images"])
    lbl_dir = Path(cfg["paths"]["raw_labels"])

    print(f"Building zero-shot predictor for organ {organ_id} ...")
    pred_zs = _load_predictor(cfg)
    pred_lora = _load_predictor(cfg, lora_path) if lora_path else None

    results = []

    for case in val_cases:
        try:
            img_path = img_dir / f"{case}.nii"
            lbl_path = lbl_dir / f"{case.replace(img_prefix, lbl_prefix)}.nii"
            frames_dir = Path(cfg["paths"]["processed"]) / case
            out_dir = Path("outputs") / case
            out_dir.mkdir(parents=True, exist_ok=True)

            if not img_path.exists() or not lbl_path.exists():
                print(f"  [skip] {case}: image or label file not found")
                continue

            if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
                print(f"  Writing PNG frames for {case} ...")
                write_frames_png(str(img_path), str(frames_dir))

            vol, _, spacing = load_volume(str(img_path))
            lbl, _, _ = load_volume(str(lbl_path))
            vol_u8 = apply_hu_window(vol)
            gt3d = (lbl == organ_id).astype(bool)

            if not gt3d.any():
                print(f"  [skip] {case}: organ {organ_id} absent")
                continue

            start_z = best_start_slice(lbl, organ_id)
            gt_slice = (lbl[:, :, start_z] == organ_id).astype(np.uint8)
            bbox = bbox_from_mask(gt_slice, pad=4)

            if bbox is None:
                print(f"  [skip] {case}: no bbox found at start_z={start_z}")
                continue

            target_hw = (vol.shape[0], vol.shape[1])

            print(f"  [{case}] zero-shot inference (start_z={start_z}) ...")
            state_zs = init_state(pred_zs, str(frames_dir))
            mask_zs = propagate_bidirectional(
                pred_zs, state_zs, start_z, bbox, target_hw=target_hw
            )
            dsc_zs = dice_score(mask_zs, gt3d)
            hd_zs = hd95(mask_zs, gt3d, spacing)
            save_overlay_gif(
                vol_u8, mask_zs, str(out_dir / f"zeroshot_{organ_id}.gif"), gt3d
            )
            _free_state(pred_zs, state_zs)

            if pred_lora is not None:
                print(f"  [{case}] LoRA inference ...")
                state_lo = init_state(pred_lora, str(frames_dir))
                mask_lo = propagate_bidirectional(
                    pred_lora, state_lo, start_z, bbox, target_hw=target_hw
                )
                dsc_lo = dice_score(mask_lo, gt3d)
                hd_lo = hd95(mask_lo, gt3d, spacing)
                save_overlay_gif(
                    vol_u8, mask_lo, str(out_dir / f"lora_{organ_id}.gif"), gt3d
                )
                _free_state(pred_lora, state_lo)
            else:
                dsc_lo = float("nan")
                hd_lo = float("nan")

            print(f"  [{case}] DSC zero-shot={dsc_zs:.3f} HD95={hd_zs:.1f}mm"
                  + (f" | DSC lora={dsc_lo:.3f} HD95={hd_lo:.1f}mm" if pred_lora is not None else ""))

            results.append(dict(
                case=case, dsc_zs=dsc_zs, hd95_zs=hd_zs,
                dsc_lora=dsc_lo, hd95_lora=hd_lo, error=None,
            ))

        except Exception as e:
            print(f"  [ERROR] {case}: {type(e).__name__}: {e} -- skipping, continuing with next case")
            results.append(dict(
                case=case, dsc_zs=float("nan"), hd95_zs=float("nan"),
                dsc_lora=float("nan"), hd95_lora=float("nan"), error=str(e),
            ))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del pred_zs
    if pred_lora is not None:
        del pred_lora
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def print_table(rows, organ_id):
    """Print a formatted DSC / HD95 summary table and return aggregated metrics.

    Specification:
    - Compute mean and std for each metric across all rows, ignoring NaN values
      (np.nanmean / np.nanstd).
    - Print the fixed-width Zero-shot vs LoRA table shown below.
    - Return a dict with organ, dsc_zs_mean/std, dsc_lora_mean/std,
      hd95_zs_mean/std, hd95_lora_mean/std, n_errors.
    """
    def _mean_std(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    dsc_zs_mean, dsc_zs_std = _mean_std("dsc_zs")
    dsc_lora_mean, dsc_lora_std = _mean_std("dsc_lora")
    hd95_zs_mean, hd95_zs_std = _mean_std("hd95_zs")
    hd95_lora_mean, hd95_lora_std = _mean_std("hd95_lora")

    n = len(rows)
    n_errors = sum(1 for r in rows if r.get("error"))

    print("=" * 60)
    print(f"Organ {organ_id} | {n} val cases" + (f" | {n_errors} errored" if n_errors else ""))
    print("=" * 60)
    print(f"{'Metric':<15}{'Zero-shot':<20}{'LoRA':<20}")
    print("-" * 60)
    print(f"{'DSC':<15}"
          f"{f'{dsc_zs_mean:.3f} \u00b1 {dsc_zs_std:.3f}':<20}"
          f"{f'{dsc_lora_mean:.3f} \u00b1 {dsc_lora_std:.3f}':<20}")
    print(f"{'HD95 (mm)':<15}"
          f"{f'{hd95_zs_mean:.1f} \u00b1 {hd95_zs_std:.1f}':<20}"
          f"{f'{hd95_lora_mean:.1f} \u00b1 {hd95_lora_std:.1f}':<20}")
    print("=" * 60)

    return dict(
        organ=organ_id,
        dsc_zs_mean=dsc_zs_mean, dsc_zs_std=dsc_zs_std,
        dsc_lora_mean=dsc_lora_mean, dsc_lora_std=dsc_lora_std,
        hd95_zs_mean=hd95_zs_mean, hd95_zs_std=hd95_zs_std,
        hd95_lora_mean=hd95_lora_mean, hd95_lora_std=hd95_lora_std,
        n_errors=n_errors,
    )

%%writefile scripts/04_evaluate.py
"""04_evaluate.py -- evaluate zero-shot vs LoRA Dice + HD95 on the val split.

Usage (run from project root):
    python scripts/04_evaluate.py --organ 6
    python scripts/04_evaluate.py --organ 6 --lora checkpoints/lora_organ6.pt

What this script does:
1. Runs bidirectional SAM2 propagation (streaming memory attention across
   z-slices) on every val-split CT volume, using the image encoder that was
   LoRA-finetuned in Week 4 (finetuning targeted the image encoder only --
   the SAM2 memory-attention module itself is left at its pretrained,
   zero-shot weights, so cross-frame propagation is unchanged).
2. Computes DSC and HD95 for both zero-shot and LoRA-tuned predictions.
3. Prints a summary table (zero-shot vs LoRA) and saves a per-case CSV.
4. Saves overlay GIFs for visual inspection.

This script is a thin CLI wrapper -- all the real work happens in:
    src/eval/evaluate.py  : _load_predictor, evaluate_organ, print_table  (Week 5)
    src/eval/metrics.py   : dice_score, hd95                              (Week 4)
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
set_seed(42)

import yaml

from src.eval.evaluate import evaluate_organ, print_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on BTCV val split")
    parser.add_argument("--organ", type=int, required=True,
                         help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--lora", default=None,
                         help="LoRA adapter .pt -- auto-detected from checkpoints/ if absent")
    parser.add_argument("--img-prefix", default="img")
    parser.add_argument("--lbl-prefix", default="label")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lora_path = args.lora
    if lora_path is None:
        auto = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
        if auto.exists():
            lora_path = str(auto)
            print(f"Auto-detected LoRA adapter: {lora_path}")
        else:
            print("No LoRA adapter found -- running zero-shot only.")

    print(f"\nEvaluating organ={args.organ}  val_cases={cfg['split']['val_cases']}\n")

    rows = evaluate_organ(cfg, args.organ, lora_path,
                           img_prefix=args.img_prefix, lbl_prefix=args.lbl_prefix)

    if not rows:
        print("No results -- check that val cases exist in the data directory.")
        return

    summary = print_table(rows, args.organ)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_organ{args.organ}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-case CSV saved -> {csv_path}")

    baselines = {6: "~0.93", 1: "~0.94", 11: "~0.72"}
    if args.organ in baselines:
        print(f"\nPublished SAM2 zero-shot baseline for organ {args.organ}: "
              f"DSC {baselines[args.organ]}")


if __name__ == "__main__":
    main()

import importlib
import src.eval.evaluate

importlib.reload(src.eval.evaluate)

from src.eval.evaluate import _load_predictor, evaluate_organ, print_table

print("All Week 5/6 modules imported successfully.")
