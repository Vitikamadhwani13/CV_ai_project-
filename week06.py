print("Multi-organ final results (Week 6)
Trains (or reuses) a LoRA checkpoint for each of liver / spleen / pancreas, evaluates all three against the val split, and builds one combined report with a comparison chart.")

import subprocess

ORGANS = {6: "liver", 1: "spleen", 11: "pancreas"}
ckpt_dir = Path(cfg["paths"]["checkpoints"])

for organ_id, organ_name in ORGANS.items():
    ckpt_path_o = ckpt_dir / f"lora_organ{organ_id}.pt"
    if ckpt_path_o.exists():
        print(f"[ok] organ {organ_id} ({organ_name}): LoRA checkpoint already present -> {ckpt_path_o}")
        continue
    print(f"[train] organ {organ_id} ({organ_name}): no checkpoint found -- running Week 4 training...")
    result = subprocess.run(
        ["python", "scripts/03_train_lora.py", "--organ", str(organ_id), "--img-prefix", IMG_PREFIX],
        check=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: training failed for organ {organ_id} -- will fall back to zero-shot only.")

for organ_id in ORGANS:
    print("\n" + "#" * 70)
    print(f"# Organ {organ_id} ({ORGANS[organ_id]})")
    print("#" * 70)
    !python scripts/04_evaluate.py --organ {organ_id} --img-prefix {IMG_PREFIX} --lbl-prefix {LBL_PREFIX}

import pandas as pd

PUBLISHED_BASELINES = {6: 0.93, 1: 0.94, 11: 0.72}  # zero-shot DSC, from the SAM2 paper / project proposal

summary_rows = []
for organ_id, organ_name in ORGANS.items():
    csv_path = Path("outputs") / f"eval_organ{organ_id}.csv"
    if not csv_path.exists():
        print(f"[skip] organ {organ_id}: no results CSV -- evaluation must have failed or found no val cases.")
        continue

    df = pd.read_csv(csv_path)
    summary_rows.append({
        "organ_id": organ_id,
        "organ": organ_name,
        "n_cases": len(df),
        "dsc_zeroshot_mean": df["dsc_zs"].mean(),
        "dsc_zeroshot_std": df["dsc_zs"].std(),
        "dsc_lora_mean": df["dsc_lora"].mean(skipna=True),
        "dsc_lora_std": df["dsc_lora"].std(skipna=True),
        "hd95_zeroshot_mean": df["hd95_zs"].mean(),
        "hd95_lora_mean": df["hd95_lora"].mean(skipna=True),
        "published_baseline_dsc": PUBLISHED_BASELINES.get(organ_id, np.nan),
    })

report_df = pd.DataFrame(summary_rows)
if not report_df.empty:
    report_df["dsc_gain_lora_vs_zeroshot"] = report_df["dsc_lora_mean"] - report_df["dsc_zeroshot_mean"]
    report_df["dsc_gap_vs_published"] = report_df["dsc_zeroshot_mean"] - report_df["published_baseline_dsc"]

report_path = Path("outputs") / "week6_final_report.csv"
report_df.to_csv(report_path, index=False)

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("=" * 90)
print("WEEK 6 -- FINAL MULTI-ORGAN RESULTS")
print("=" * 90)
display(report_df)
print(f"\nSaved -> {report_path}")

import matplotlib.pyplot as plt

if not report_df.empty:
    organs = report_df["organ"]
    x = np.arange(len(organs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, report_df["dsc_zeroshot_mean"], width, label="Zero-shot (ours)")
    ax.bar(x, report_df["dsc_lora_mean"], width, label="LoRA-finetuned (ours)")
    ax.bar(x + width, report_df["published_baseline_dsc"], width, label="Published baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(organs)
    ax.set_ylabel("Dice Similarity Coefficient (DSC)")
    ax.set_ylim(0, 1)
    ax.set_title("SAM2 Volumetric Segmentation -- DSC by Organ")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/week6_dsc_comparison.png", dpi=150)
    plt.show()
else:
    print("No results to plot yet -- check the evaluation runs above.")

from IPython.display import Image as IPyImage, display

for organ_id, organ_name in ORGANS.items():
    csv_path = Path("outputs") / f"eval_organ{organ_id}.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    first_case = df.iloc[0]["case"]

    zs_gif = Path("outputs") / first_case / f"zeroshot_{organ_id}.gif"
    lo_gif = Path("outputs") / first_case / f"lora_{organ_id}.gif"

    print(f"\n--- {organ_name} (organ {organ_id}), case {first_case} ---")
    if zs_gif.exists():
        print("Zero-shot:")
        display(IPyImage(filename=str(zs_gif)))
    if lo_gif.exists():
        print("LoRA-finetuned:")
        display(IPyImage(filename=str(lo_gif)))

def final_checklist():
    report = {
        "Week 1 -- NIfTI I/O + HU windowing": True,
        "Week 2 -- Prompts, PNG frames, BTCVSliceDataset": True,
        "Week 3 -- SAM2 bidirectional propagation engine": True,
        "Week 4 -- LoRA fine-tuning (image encoder)": lora_ckpt.exists(),
        "Week 5 -- evaluate.py (zero-shot vs LoRA harness)": True,
        "Week 6 -- LoRA checkpoints for liver/spleen/pancreas":
            all((ckpt_dir / f"lora_organ{o}.pt").exists() for o in ORGANS),
        "Week 6 -- Per-organ eval CSVs saved":
            all((Path("outputs") / f"eval_organ{o}.csv").exists() for o in ORGANS),
        "Week 6 -- Combined week6_final_report.csv saved": Path("outputs/week6_final_report.csv").exists(),
        "Week 6 -- DSC comparison chart saved": Path("outputs/week6_dsc_comparison.png").exists(),
    }
    print("=" * 70)
    print("FINAL COMPLETION REPORT -- WEEKS 1-6")
    print("=" * 70)
    for item, status in report.items():
        print(f"{item:52} : {status}")
    print("=" * 70)

final_checklist()

print("Reproducibility -- Environment Snapshot")

import subprocess
from pathlib import Path

key_packages = ["torch", "nibabel", "monai", "SimpleITK", "peft", "hydra-core",
                 "iopath", "imageio", "einops", "numpy", "scipy", "pandas"]

freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
pinned = [line for line in freeze.splitlines()
          if any(line.lower().startswith(p.lower() + "==") for p in key_packages)]

Path("requirements_frozen.txt").write_text("\n".join(sorted(pinned)) + "\n")

print("Key package versions used in this run:")
for line in sorted(pinned):
    print(" ", line)
print("\nFull pinned list saved -> requirements_frozen.txt")

try:
    sam2_commit = subprocess.run(
        ["git", "-C", "sam2", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"SAM2 repo commit: {sam2_commit}")
    Path("requirements_frozen.txt").open("a").write(f"# sam2 @ {sam2_commit}\n")
except Exception as e:
    print("Could not resolve SAM2 commit hash:", e)
