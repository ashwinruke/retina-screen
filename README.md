# Retinal DR Screening

Diabetic retinopathy grading from fundus photographs, built to generalise
beyond the dataset it was trained on:

- **Pretrained** on Kaggle EyePACS 2015 (~88k images)
- **Fine-tuned** on APTOS 2019 (3.6k images, Aravind Eye Hospital)
- **Externally tested** on IDRiD (516 images, a different Indian clinic and camera, never seen in training)
- **Explained** with Grad-CAM heatmaps
- **Demo** on Hugging Face Spaces

> Research project, not a medical device.

**Status:** APTOS baseline, external test on IDRiD and live demo done (results
below). Next: EyePACS pretraining.

**Live demo:** [huggingface.co/spaces/ashwinruke/retina-dr-screening](https://huggingface.co/spaces/ashwinruke/retina-dr-screening)
runs the model in your browser, so photos are never uploaded.

## Results

All numbers below come from one model: EfficientNet-B3 at 512 px, ImageNet
weights, trained 15 epochs on APTOS on one Kaggle T4 (~18 min). Cut points
between grades were fit on APTOS validation data and then frozen.

| | APTOS test | IDRiD, all images | IDRiD official test |
|---|---|---|---|
| Images | 523 | 516 | 103 |
| Source | same as training | **new clinic and camera** | subset of the 516 |
| Quadratic weighted kappa | **0.920** | **0.787** | 0.682 |
| Accuracy (5 grades) | 0.820 | 0.570 | 0.495 |
| Referable DR: ROC-AUC | **0.985** | **0.960** | 0.904 |
| Referable DR: sensitivity / specificity | 0.951 / 0.903 | 0.830 / 0.964 | 0.813 / 0.872 |

Referable DR = grade ≥ 2 (moderate or worse), the threshold for referral to an
eye specialist. The APTOS test split is 15% of the APTOS training set held out
before training, so it isn't comparable to the APTOS Kaggle leaderboard.

### Training and APTOS test

![Training curves](docs/figures/mvp_training.png)

| | |
|---|---|
| ![APTOS confusion matrix](docs/figures/mvp_confusion.png) | ![APTOS referable DR ROC](docs/figures/mvp_roc.png) |

### External validation on IDRiD

IDRiD was never used for training, model selection or tuning.

| | |
|---|---|
| ![IDRiD confusion matrix](docs/figures/idrid_confusion.png) | ![IDRiD referable DR ROC](docs/figures/idrid_roc.png) |

What it shows:

- **Screening transfers; exact grading drops.** Referable-DR AUC only falls from
  0.985 to 0.960 on a new clinic and camera, but QWK falls from 0.920 to 0.787.
  The model still tells "refer" from "don't refer" well; it's less sure of the
  exact grade.
- **Severe disease is under-graded.** 58 of 93 severe (grade 3) eyes are
  called moderate, so only 9% are graded correctly. They are still flagged as
  referable, but without the urgency they need. Severe is rare in APTOS (5% of
  images) and common in IDRiD (18%).
- **Tuned cut points don't transfer.** The default cut points (0.5 / 1.5 / 2.5 / 3.5)
  score better on IDRiD than the tuned ones (QWK 0.797 vs 0.787), and severe
  recall rises from 9% to 46%. Tuning fit a severe band only 0.19 wide
  (2.79–2.97) to the 25 severe eyes in APTOS validation; with so few examples
  the fit is unstable, and it fails under a change of population.
- **The 103-image official test set is small** (5 mild, 13 proliferative eyes),
  so its numbers swing by several points with a handful of predictions. The
  516-image figure is the headline.

### Full comparison

_Filled in as each step lands._

| Run | APTOS test QWK | IDRiD QWK | IDRiD referable AUC | IDRiD sens / spec |
|---|---|---|---|---|
| A0 · ImageNet → APTOS | 0.920 | 0.787 | 0.960 | 0.830 / 0.964 |
| **A1 · EyePACS → APTOS (main)** | | | | |
| A2 · A1 without Ben Graham | | | | |
| A3 · A1 with classification head | | | | |
| A4 · A1 + IDRiD fine-tune (IDRiD test only) | — | | | |

QWK = quadratic weighted kappa. IDRiD columns use all 516 images, except A4,
which trains on IDRiD's 413 training images and reports on its 103 test images
(A0 on those 103: QWK 0.682).

## Data cleaning

APTOS contains the same photo several times, sometimes with different grades.
Copies that land on both sides of a train/test split inflate scores, and
conflicting labels are noise. Every image gets two fingerprints: its fine
detail (vessels, lesions) and its overall layout (optic disc position, which
eye). Two images count as copies only when both match.

| | Images |
|---|---|
| APTOS labelled images | 3,662 |
| Duplicate groups found | 131 |
| Extra copies removed (same grade, one kept) | 97 |
| Removed: copies with conflicting grades | 79 |
| **Used** | **3,486** |

Split 70 / 15 / 15, stratified by grade. Every flagged group is listed in
`data/manifests/aptos_duplicates.csv` for review.

## How it works

```
fundus photo → crop black border → pad to square → 512 px → Ben Graham normalisation
            → EfficientNet-B3 → GeM pooling → 1 regression output (severity 0–4)
            → thresholds tuned on val → grade 0–4 + Grad-CAM heatmap
```

Design decisions and their reasons are in [docs/PLAN.md](docs/PLAN.md):
why regression instead of 5-way classification, why patient-level splits on
EyePACS, why APTOS is deduplicated, and why IDRiD is held out entirely.

## Quickstart

### Dry run on synthetic data (CPU, ~1 minute)

```bash
pip install -r requirements.txt
python scripts/make_synthetic.py --out data/synthetic
python scripts/prepare_data.py --aptos data/synthetic/aptos --eyepacs data/synthetic/eyepacs \
    --idrid data/synthetic/idrid --out-dir data/manifests
python scripts/cache_images.py --manifest data/manifests/aptos.csv --out-dir data/synthetic_cache \
    --config configs/smoke.yaml --workers 2
python scripts/train.py --config configs/smoke.yaml
pytest
```

### Real training on Kaggle

Open [`notebooks/kaggle_pipeline.ipynb`](notebooks/kaggle_pipeline.ipynb) on
Kaggle, attach the three datasets, turn on GPU T4 x2 and Internet, and run it.
The steps it runs:

```bash
python scripts/prepare_data.py --aptos ... --eyepacs ... --idrid ... --out-dir data/manifests
python scripts/cache_images.py --manifest data/manifests/eyepacs.csv --out-dir cache --config configs/pretrain_eyepacs.yaml
python scripts/train.py --config configs/pretrain_eyepacs.yaml        # stage 1 (resumable: --resume .../last.pt)
python scripts/train.py --config configs/finetune_aptos.yaml          # stage 2
python scripts/artifacts.py push --tag <name> --run <run folders>     # save results (Kaggle wipes outputs)
python scripts/evaluate.py --checkpoint outputs/finetune_aptos/best.pt --manifest cache/aptos.csv --splits test \
    --fit-manifest cache/aptos.csv --fit-splits val --export outputs/model.pt
python scripts/evaluate.py --checkpoint outputs/model.pt --manifest cache/idrid.csv --splits all --name idrid
```

Any config value can be overridden on the command line, e.g.
`python scripts/train.py --config configs/finetune_aptos.yaml model.backbone=convnext_tiny train.epochs=20`.

### Demo

The live demo is a static page (`web/`) that runs the model in the visitor's
browser with ONNX Runtime Web, so photos never leave their device and it
needs no server. Everything after crop-and-resize (colour normalisation, the
network, flip TTA and Grad-CAM) is inside the exported ONNX graph, and the
export checks itself against the PyTorch pipeline.

```bash
pip install onnx onnxscript onnxruntime
python scripts/export_onnx.py --model outputs/model.pt --out-dir web \
    --check-images cache/aptos.csv --report outputs/eval/mvp_aptos_test.json
python -m http.server -d web 8000                  # preview at http://localhost:8000
HF_TOKEN=... python scripts/deploy_space.py --space <user>/retina-dr-screening
```

`app/app.py` is the same demo as a server-side Gradio app
(`MODEL_PATH=outputs/model.pt python app/app.py`).

## Repo layout

```
configs/      one YAML per experiment (stage 1, stage 2, baseline, IDRiD adaptation, smoke test)
src/retina/   preprocess · manifests · data · model · engine (train/predict) · metrics · gradcam
scripts/      prepare_data · cache_images · train · evaluate · plot_results · export_onnx
              deploy_space · benchmark_gpu · make_synthetic · build_space.sh
web/          in-browser demo (static page + ONNX model)
app/          the same demo as a Gradio app
notebooks/    Kaggle notebook running the full pipeline
tests/        unit tests + end-to-end test on synthetic data
docs/PLAN.md  project plan, design decisions, milestones
docs/figures/ result figures used in this README
```

## Data

The datasets are not included. Get them from their sources and follow their terms:

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (Kaggle competition)
- [Diabetic Retinopathy Detection (EyePACS)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) (Kaggle competition)
- [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) (IEEE DataPort, CC BY 4.0)

## Limitations

- One image per eye, one model; no ensembling.
- Grades only DR; doesn't detect glaucoma, AMD or other conditions, and doesn't
  judge image quality.
- EyePACS labels are noisy; APTOS and IDRiD are small. The IDRiD result is the
  honest estimate of performance on a new clinic, and it is lower than APTOS test.
