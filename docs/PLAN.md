# Retinal Disease Screening — Project Plan

Automated diabetic retinopathy (DR) grading from colour fundus photographs, with
visual explanations and a public demo. The model is pretrained on a large noisy
dataset (EyePACS), fine-tuned on a cleaner one (APTOS), and tested on a dataset
it never sees during training (IDRiD) to show how it holds up on a new hospital,
camera and population.

## 1. Problem definition

**Task.** Given one fundus image, predict the DR severity grade on the
International Clinical DR scale:

| Grade | Meaning | Clinical action |
|---|---|---|
| 0 | No DR | Rescreen in 12 months |
| 1 | Mild NPDR | Rescreen in 6–12 months |
| 2 | Moderate NPDR | **Refer** |
| 3 | Severe NPDR | **Refer (urgent)** |
| 4 | Proliferative DR | **Refer (urgent)** |

Two outputs are reported:

1. **5-class grade**, scored with quadratic weighted kappa (QWK). QWK punishes
   predicting 4 for a 0 far more than predicting 1 for a 0, which matches how
   wrong each mistake actually is.
2. **Referable DR (grade ≥ 2) yes/no**, scored with ROC-AUC, sensitivity and
   specificity. This is the number a screening programme cares about.

The grades are ordered, so the default head is a **single regression output**
(predict a continuous score, then cut it into grades with thresholds tuned on
the validation set). This consistently beat plain 5-way classification in the
APTOS 2019 competition. A classification head is available as an ablation.

## 2. Datasets and their roles

| Dataset | Images | Labels | Role |
|---|---|---|---|
| **Kaggle EyePACS 2015** | ~35k train + ~53k test (test labels released after the competition) | 0–4 grade per eye | **Stage 1 pretraining.** Large but noisy; many images over/under-exposed or out of focus. |
| **APTOS 2019** | 3,662 labelled | 0–4 grade | **Stage 2 fine-tuning** + in-distribution validation and test. Images from Aravind Eye Hospital, India. |
| **IDRiD** (grading subset) | 413 train + 103 test | 0–4 DR grade (+ macular-edema risk) | **External test set.** Never used for training or threshold tuning in the main experiment. |

### Splits

- **EyePACS**: 95 / 5 train / val, **grouped by patient** (`10_left` and
  `10_right` are the same person). Splitting by image would leak one eye of a
  patient into validation and inflate the score.
- **APTOS**: 70 / 15 / 15 train / val / test, stratified by grade, after
  removing duplicates (see pitfalls).
- **IDRiD**: kept whole as an external test. As an optional extra experiment
  (Stage 3), fine-tune on IDRiD train and report on IDRiD test, to measure how
  much a small amount of local data closes the domain gap.

### Getting the data on Kaggle

Attach all three as **Inputs** in a Kaggle notebook (search the dataset names in
"Add Input"). For EyePACS, use a **pre-resized** copy if one is available — the
original release is ~80 GB of 3–5 megapixel JPEGs, and decoding those every
epoch would make pretraining CPU-bound. `scripts/prepare_data.py` finds the
label CSVs and images by searching the attached folders, so the exact folder
names don't matter.

## 3. Pipeline

```
raw images ──prepare_data──▶ manifests (path, grade, patient, split)
           ──cache_images──▶ cropped, 512 px, Ben-Graham-normalised JPEGs
           ──train (stage 1: EyePACS)──▶ pretrained checkpoint
           ──train (stage 2: APTOS, init from stage 1)──▶ fine-tuned checkpoint
           ──evaluate──▶ thresholds fit on APTOS val → report on APTOS test + IDRiD
           ──export──▶ model.pt (weights + preprocessing + thresholds) ──▶ Gradio app
```

### 3.1 Preprocessing (done once, cached to disk)

1. **Crop to the fundus**: drop the black border by thresholding brightness,
   then pad to a square so the circular retina is centred.
2. **Resize** to 512 × 512 (`INTER_AREA`).
3. **Ben Graham normalisation**: `4·img − 4·GaussianBlur(img, σ=size/30) + 128`.
   This removes slow lighting variation between cameras and makes
   microaneurysms, haemorrhages and exudates stand out. It was the standard
   trick in both Kaggle competitions and matters most for cross-dataset
   generalisation.
4. **Circular mask** at 95 % radius to remove the bright ring artefact the blur
   subtraction creates at the edge.

The same function runs inside the demo, and the settings are saved in every
checkpoint, so training and inference can't drift apart.

### 3.2 Augmentation

Fundus images have no "up", so rotation is safe at any angle. Training uses
random rotation (±180°), horizontal/vertical flips, small scale changes (±10 %),
and mild brightness/contrast jitter. Validation and test use resize only; test
time uses horizontal-flip averaging (TTA).

### 3.3 Model

- **Backbone**: `efficientnet_b3` from `timm`, ImageNet-pretrained. Good
  accuracy per FLOP at 512 px and fits a Kaggle T4 at batch 16 with mixed
  precision. Swappable via config (`resnet50`, `convnext_tiny`, `efficientnet_b4`…).
- **Pooling**: GeM (generalised mean) pooling, which emphasises small bright
  activations — useful because lesions are small.
- **Head**: dropout → linear → 1 output (regression), SmoothL1 loss.
- **Imbalance**: EyePACS is ~73 % grade 0. A weighted sampler with weights
  ∝ 1/√(class count) softens this without massively oversampling the few grade
  4 images.

### 3.4 Training schedule

| | Stage 1 — EyePACS | Stage 2 — APTOS |
|---|---|---|
| Init | ImageNet | Stage 1 best checkpoint |
| Epochs | 12 | 15 |
| LR (head / backbone) | 1e-3 / 3e-4 | 3e-4 / 1e-4 |
| Schedule | linear warmup 1 epoch → cosine | same |
| Freeze backbone | first 1 epoch | none |
| Batch / precision | 16 / AMP fp16 | 16 / AMP fp16 |
| Model selection | val QWK | val QWK |

Rough Kaggle budget (T4, 512 px, B3): stage 1 ≈ 25–35 min/epoch on ~83k
images → split across 2–3 sessions using `--resume`. Stage 2 ≈ 1–2 min/epoch.
Kaggle gives ~30 GPU hours per week and 12 hours per session, so the whole
project fits in one week of quota.

### 3.5 Thresholds

Regression scores are converted to grades with four cut points, starting at
`[0.5, 1.5, 2.5, 3.5]`. Coordinate search on the **APTOS validation set** tunes
them to maximise QWK. They are then frozen and applied unchanged to APTOS test
and IDRiD. Tuning thresholds on the test set would be leakage.

## 4. Evaluation

For each test set (APTOS test, IDRiD train+test):

- QWK, accuracy, confusion matrix, per-grade recall
- Referable DR: ROC-AUC, sensitivity and specificity at grade ≥ 2
- Results with default vs tuned thresholds, and with/without TTA

### Ablations (one row each in the README results table)

| # | Change | Question it answers |
|---|---|---|
| A0 | ImageNet → APTOS only (no EyePACS) | How much does pretraining on EyePACS help? |
| A1 | Full pipeline | Main result |
| A2 | No Ben Graham preprocessing | Does normalisation help on an external dataset? |
| A3 | Classification head instead of regression | Does treating grades as ordered help QWK? |
| A4 | + Stage 3 fine-tune on IDRiD train | How much local data closes the domain gap? |

The A1 vs A0 and A1 vs A2 gaps should be larger on IDRiD than on APTOS test.
That is the result worth highlighting.

## 5. Explainability

Grad-CAM on the last backbone feature map. For the regression head, the target
is the predicted score itself, so the heatmap shows which regions pushed the
score up. Shown as an overlay on the cropped (non-normalised) fundus in the
demo.

**Sanity check worth putting in the README**: IDRiD ships pixel-level lesion
masks (microaneurysms, haemorrhages, hard and soft exudates) for 81 images.
Measuring how much of the Grad-CAM mass falls inside those masks (a "pointing
game" score) turns "the heatmaps look plausible" into a number.

## 6. Demo (Gradio on Hugging Face Spaces)

- Upload a fundus photo → predicted grade, continuous severity score, referable
  yes/no, and the Grad-CAM overlay.
- Loads `model.pt` from the Space repo or from a Hugging Face model repo
  (`HF_MODEL_REPO` env var).
- Runs on the free CPU Space: B3 at 512 px is ~1–2 s per image.
- Prominent disclaimer: research demo, not a medical device.

## 7. Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Data plumbing | Manifests for all 3 datasets build on Kaggle; cached images look right in a grid |
| M2 | APTOS-only baseline (A0) | Val QWK ≥ 0.88 |
| M3 | EyePACS pretraining | Val QWK on EyePACS ≥ 0.75 (labels are noisy; this is normal) |
| M4 | Fine-tune (A1) | APTOS test QWK ≥ 0.90; IDRiD QWK reported |
| M5 | Ablations A2–A4 | Results table filled in |
| M6 | Explainability | Grad-CAM grid + pointing-game score on IDRiD masks |
| M7 | Demo | Live HF Space linked from README |
| M8 | Write-up | README with results, figures, limitations |

The QWK targets are typical for a single B3 at 512 px; top APTOS solutions
reached ~0.93 on the private leaderboard with large ensembles.

## 8. Pitfalls to handle

- **APTOS duplicates.** The training set contains near-identical images, some
  with *different* labels. `prepare_data.py --dedup` hashes every image, drops
  groups with conflicting labels, and keeps one copy of the rest. Without this,
  copies land in both train and test and inflate the score.
- **EyePACS label noise.** Expect lower QWK there; it is a pretraining signal,
  not the headline metric.
- **Ungradable images.** EyePACS has some nearly black or blurry images. The
  crop step leaves an image untouched if it can't find a fundus; consider
  dropping images whose mean brightness is very low.
- **Patient leakage.** Always split EyePACS by patient.
- **Threshold leakage.** Fit thresholds on validation only.
- **Camera and population shift.** IDRiD is a different clinic and camera.
  Report the drop honestly; it's more convincing than hiding it.

## 9. Repo layout

```
retina-screen/
├── configs/               # pretrain_eyepacs.yaml, finetune_aptos.yaml, ...
├── src/retina/            # preprocessing, datasets, model, training, metrics, Grad-CAM
├── scripts/               # prepare_data, cache_images, train, evaluate, make_synthetic
├── app/                   # Gradio demo for Hugging Face Spaces
├── notebooks/             # Kaggle notebook that runs the whole pipeline
├── tests/                 # unit tests + end-to-end smoke test on synthetic data
└── docs/PLAN.md           # this file
```

## 10. Licensing note

The datasets have their own terms (Kaggle competition rules for EyePACS and
APTOS; CC BY 4.0 for IDRiD). Don't commit images to the repo or bundle them in
the Space — link to the sources instead. Check the EyePACS/APTOS competition
rules before publishing model weights publicly.
