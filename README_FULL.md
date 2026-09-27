# Volumetric Medical Segmentation via SAM 2 Streaming Memory Attention

A 3D abdominal organ segmentation pipeline built by adapting Meta's Segment Anything Model 2 (SAM 2) — designed for temporal video tracking — to treat the z-axis of CT volumes as a sequence of video frames.

**Project by IITG.AI · BTCV Abdominal Organ Segmentation Dataset**

## 1. The Problem

Most medical segmentation models work slice by slice in 2D. That's a real limitation for CT and MRI volumes, because nothing forces the model to keep an organ's shape consistent as it moves from one slice to the next — you can end up with a liver that's smooth in one slice and jagged or disconnected two slices later, purely because each slice was segmented in isolation.

SAM 2 already solves a version of this problem, just not for medical imaging. It was built by Meta to track objects through video, using a "streaming memory attention" mechanism that lets it remember what an object looked like in earlier frames and carry that understanding forward.

The idea behind this project is simple to state and harder to execute: if a CT volume's z-axis is treated as a sequence of frames the same way a video's time axis is, SAM 2's memory attention should be able to propagate a single prompt — one bounding box on one slice — across an entire 3D organ, keeping it anatomically coherent the whole way through.

We tested this on the BTCV (Beyond the Cranial Vault) dataset, a standard 30-case abdominal CT benchmark with 13 annotated organs, focusing on three: liver, spleen, and pancreas.

## 2. Approach, In Short

1. Convert each CT volume into a folder of PNG "frames," one per axial slice, after windowing the Hounsfield units to a soft-tissue range.
2. Feed those frames into SAM 2's video predictor and drop a single bounding-box prompt on the slice with the largest cross-section of the target organ.
3. Propagate bidirectionally — forward from that slice to the end of the volume, backward to the start — letting SAM 2's memory attention carry the mask through every slice in between.
4. Fine-tune SAM 2's image encoder with LoRA on BTCV's training split, so the visual features it extracts are tuned to CT anatomy instead of natural images, while leaving the memory-attention mechanism itself untouched.
5. Evaluate zero-shot vs. LoRA-finetuned performance with volumetric Dice and 95th-percentile Hausdorff Distance, against the published SAM 2 baselines for this dataset.

## 3. Tech Stack

- **Segmentation:** SAM 2.1 (Hiera-Tiny) from Meta AI for medical organ segmentation.
- **Fine-tuning:** LoRA with rank 8 and alpha 16 using `peft` for efficient model adaptation.
- **Medical Data:** `nibabel` for loading and processing 3D NIfTI medical volumes.
- **Evaluation:** Custom Dice score for segmentation overlap and MONAI Hausdorff Distance for boundary accuracy.
- **Training:** PyTorch with bfloat16 mixed precision and gradient accumulation for efficient GPU training.
- **Configuration:** A single `configs/default.yaml` file manages settings across the complete pipeline.
- **Environment:** Kaggle notebooks with T4/P100 GPUs for model training and experimentation.

## 4. Repository Structure

- **`src/data/`** – Handles medical data preparation. It includes NIfTI loading, preprocessing, prompt generation, conversion of 3D volumes into 2D image frames, and dataset creation for LoRA training.
- **`src/engine/`** – Contains the SAM 2 inference pipeline, including predictor setup and bidirectional mask propagation across image slices.
- **`src/train/`** – Contains the training components, including Dice and BCE losses, LoRA integration, and the PyTorch training loop with mixed precision and gradient accumulation.
- **`src/eval/`** – Handles model evaluation using metrics such as Dice score and HD95, and provides the evaluation pipeline for comparing zero-shot SAM 2 with the LoRA fine-tuned model.
- **`src/utils/`** – Provides utility functions for reproducibility and visualization, including random seed management and segmentation overlay generation.
- **`scripts/`** – Contains the main executable scripts for preprocessing, inference, LoRA training, and evaluation.
- **`configs/`** – Stores the central `default.yaml` configuration file containing project settings.
- **`checkpoints/` and `outputs/`** – Store trained LoRA adapters, evaluation results, per-case CSV files, visualization GIFs, and final reports.
- **`tests/`** – Contains unit tests for checking important components such as Dice and HD95 metric calculations.

## 5. Week-by-Week Breakdown

### Week 0 — Dataset Acquisition & Environment Bring-Up

Before any modeling could happen, the BTCV dataset (controlled-access, hosted on Synapse) had to actually get into the working environment reliably — which turned out to be its own small project. BTCV isn't a `wget`-and-go dataset: it needs a Synapse account and an accepted data-use agreement, and the raw download comes as nested zips that don't match any single expected folder layout.

What's unique here: the setup notebook doesn't assume one fixed way of getting the data in. It auto-detects `imagesTr`/`labelsTr` whether they arrive as plain folders, zipped, or nested inside a wrapper folder from the original download; it cross-references `dataset_0.json` to determine the *actual* train/validation split rather than assuming a round number of cases (the real split turned out to be 24 training + 6 validation cases, with a gap at case IDs 0011–0020 — a detail that would have silently produced false "missing file" warnings if hardcoded). It also validates every file pair actually exists and does a live NIfTI load-and-inspect on a sample volume before letting the rest of the pipeline proceed, so a corrupted or incomplete upload gets caught in Week 0 instead of three weeks later during evaluation.

### Week 1 — Core NIfTI I/O & Visualization

This week built the foundational utilities every later stage depends on: loading a NIfTI volume and reorienting it to a canonical axis order, windowing raw Hounsfield Units down to a soft-tissue range SAM 2's ViT encoder can actually work with, and converting single-channel CT slices into the 3-channel RGB format a natural-image model expects.

What's unique here: CT volumes are anisotropic — in-plane resolution is roughly 1mm, but slice spacing (z) is typically 3–5mm, sometimes more. That's easy to overlook and easy to get wrong downstream (a bounding box padded in pixels behaves very differently at 1mm spacing than at 5mm spacing). The loader surfaces spacing explicitly every time a volume is loaded, and an isotropic resampling utility was built (using nearest-neighbor for label masks specifically, to avoid interpolation blurring organ boundaries into fractional label values) so later stages have the option to normalize spacing before training.

### Week 2 — Preprocessing Pipeline

This is where volumes actually become something SAM 2 can consume: a folder of sequentially numbered PNG frames, plus the prompt-generation logic that finds where to place the first bounding box.

What's unique here: the `best_start_slice` scans every z-slice for the one with the largest cross-sectional area of the target organ, since that's the slice where a bounding box prompt gives SAM 2 the most reliable starting shape to propagate outward from. The bounding-box extraction pads a fixed margin around the tight organ boundary (clamped to image edges) rather than using the raw tight box, which matters because SAM 2's box prompts are sensitive to how tightly they hug the actual object — a completely tight box tends to bias the mask inward.

### Week 3 — SAM 2 Inference Engine

This is the heart of the project: wiring up SAM 2's video predictor and implementing bidirectional propagation.

What's unique here: SAM 2 was designed to propagate forward through video, not outward in both directions from a middle frame. Since the prompt slice is usually somewhere in the interior of the organ (not the first slice), a genuinely bidirectional propagation loop was implemented — one forward pass from the prompt slice to the end of the volume, and a separate backward pass from the prompt slice to the start, merging both into a single 3D mask without duplicating or overwriting the prompt slice itself. Each per-slice mask comes back from SAM 2 as a GPU-resident logit tensor at the model's working resolution, not the volume's native resolution, so there's also a resize-and-threshold step (nearest-neighbor, to preserve hard mask boundaries) reconciling the two before stacking everything into a single `(H, W, Z)` volume.

### Week 4 — LoRA Fine-Tuning

Zero-shot SAM 2 has never seen a CT scan during pretraining — it was trained on natural images and video. This week's job was adapting its visual features to CT anatomy without touching the (already working) memory-attention propagation logic, and without the cost of full fine-tuning a foundation model.

What's unique here: LoRA was deliberately scoped to the image encoder's fused QKV projections only — the memory-attention module that does the actual cross-slice propagation was left frozen. That's a specific architectural choice, not an accident: it isolates *what* the model sees per slice from *how* it carries that understanding across slices, which makes it possible to attribute a change in performance specifically to feature quality rather than a confound with the propagation mechanism. Two independent LoRA implementations were built side by side — a `peft`-based standard adapter, and a custom Q/V-only variant that explicitly zeros the LoRA delta on the K (key) slice of the fused QKV projection — because gradients don't flow cleanly through SAM 2's `SAM2ImagePredictor.set_image()` (it runs under `torch.no_grad()` internally), so a custom forward path through the model's internal encoder/decoder was implemented specifically to keep LoRA's gradients attached during training. Training itself runs in bfloat16 with gradient accumulation, sized to fit a single mid-range GPU.

### Week 5 — Evaluation Harness

This week built the infrastructure to answer the actual research question cleanly: does LoRA fine-tuning help, and by how much, per organ?

**What's unique here:** the harness runs zero-shot and LoRA-finetuned inference through the *identical* propagation code path, on the *identical* validation cases, so any difference in Dice/HD95 is attributable to the fine-tuned features and nothing else. Hausdorff Distance specifically needed care — MONAI's `compute_hausdorff_distance` defaults to excluding a "background" channel, which is the correct behavior for multi-class one-hot masks but would silently corrupt results on a single-channel binary mask if applied carelessly. This was verified empirically (not assumed) before shipping: constructing synthetic masks and confirming MONAI's channel-exclusion logic only activates with two or more channels, so the single-channel binary case was already being scored correctly.

### Week 6 — Multi-Organ Final Report

The last stage runs everything built in Weeks 1–5 to completion across all three target organs (liver, spleen, pancreas) and compiles one final, decision-ready report.

What's unique here: it auto-detects whether a LoRA checkpoint already exists for a given organ and only trains the ones that are missing, so re-running the notebook doesn't waste GPU time retraining organs already done. Every case in the evaluation loop is wrapped in isolated error handling — a single corrupted volume or an out-of-memory hiccup on one case gets logged and skipped rather than aborting the entire organ's evaluation run, which matters a lot for something meant to run unattended for an hour or more. The final report cross-references measured zero-shot Dice against the *published* SAM 2 baselines from the literature (liver ≈0.93, spleen ≈0.94, pancreas ≈0.72), so the numbers aren't just internally consistent — they're checked against an external reference point.

## Known Limitations

- LoRA fine-tunes the image encoder only; SAM 2's memory-attention module stays at pretrained weights, so propagation quality has a ceiling set by that untouched component.
- Evaluated on the Hiera-Tiny SAM 2 variant for speed and memory footprint — larger variants (Small, Base+) would likely improve absolute Dice at the cost of training/inference time.
- The training split (24 cases) is small by deep-learning standards; results should be read as a feasibility demonstration on this architecture, not a claim of state-of-the-art performance.

## Acknowledgments

- SAM 2 — Meta AI, [Segment Anything 2](https://github.com/facebookresearch/sam2)
- BTCV Dataset — Multi-Atlas Labeling Beyond the Cranial Vault, Vanderbilt University
- MONAI — Hausdorff Distance implementation
