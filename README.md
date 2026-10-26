# Retinal DR Screening

Diabetic retinopathy grading from fundus photographs, built to generalise
beyond the dataset it was trained on:

- **Pretrained** on Kaggle EyePACS 2015 (~88k images)
- **Fine-tuned** on APTOS 2019 (3.6k images, Aravind Eye Hospital)
- **Externally tested** on IDRiD (516 images, a different Indian clinic and camera, never seen in training)
- **Explained** with Grad-CAM heatmaps
- **Demo** on Hugging Face Spaces

> Research project, not a medical device.

<!-- Replace with your Space link once deployed -->
**Live demo:** _coming soon_

## Results

_Fill in after training. Thresholds fit on APTOS val, then frozen._

| Run | APTOS test QWK | IDRiD QWK | IDRiD referable AUC | IDRiD sens / spec |
|---|---|---|---|---|
| A0 · ImageNet → APTOS | | | | |
| **A1 · EyePACS → APTOS (main)** | | | | |
| A2 · A1 without Ben Graham | | | | |
| A3 · A1 with classification head | | | | |
| A4 · A1 + IDRiD fine-tune (IDRiD test only) | — | | | |

QWK = quadratic weighted kappa. Referable DR = grade ≥ 2.

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
python scripts/evaluate.py --checkpoint outputs/finetune_aptos/best.pt --manifest cache/aptos.csv --splits test \
    --fit-manifest cache/aptos.csv --fit-splits val --export outputs/model.pt
python scripts/evaluate.py --checkpoint outputs/model.pt --manifest cache/idrid.csv --splits all --name idrid
```

Any config value can be overridden on the command line, e.g.
`python scripts/train.py --config configs/finetune_aptos.yaml model.backbone=convnext_tiny train.epochs=20`.

### Demo

```bash
MODEL_PATH=outputs/model.pt python app/app.py               # local
bash scripts/build_space.sh outputs/model.pt space/          # folder to push to a HF Space
```

## Repo layout

```
configs/      one YAML per experiment (stage 1, stage 2, baseline, IDRiD adaptation, smoke test)
src/retina/   preprocess · manifests · data · model · engine (train/predict) · metrics · gradcam
scripts/      prepare_data · cache_images · train · evaluate · make_synthetic · build_space.sh
app/          Gradio app for Hugging Face Spaces
notebooks/    Kaggle notebook running the full pipeline
tests/        unit tests + end-to-end test on synthetic data
docs/PLAN.md  project plan, design decisions, milestones
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
