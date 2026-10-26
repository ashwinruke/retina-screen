"""Gradio demo: upload a fundus photo, get a DR grade and a Grad-CAM heatmap.

Runs locally (`python app/app.py`) or as a Hugging Face Space (see
scripts/build_space.sh). The model file is looked up in this order:
  1. $MODEL_PATH (default: model.pt next to this file)
  2. model.pt downloaded from the Hugging Face model repo in $HF_MODEL_REPO
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "src"):  # Space layout, then repo layout
    if (candidate / "retina").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from retina.data import build_transforms  # noqa: E402
from retina.engine import load_for_inference  # noqa: E402
from retina.gradcam import gradcam, overlay  # noqa: E402
from retina.metrics import GRADE_NAMES, REFERABLE_GRADE, scores_to_grades  # noqa: E402
from retina.preprocess import background_value, crop_and_resize, preprocess_from_config  # noqa: E402

ADVICE = [
    "No referral needed. Routine rescreening in 12 months.",
    "No referral needed. Rescreen in 6-12 months.",
    "Referable. Refer to an ophthalmologist.",
    "Referable. Urgent referral to an ophthalmologist.",
    "Referable. Urgent referral to an ophthalmologist.",
]

DISCLAIMER = (
    "**Research demo, not a medical device.** Predictions come from a model trained on "
    "public datasets and have not been clinically validated. Do not use them for diagnosis."
)


def _resolve_model_path() -> Path:
    path = Path(os.environ.get("MODEL_PATH", HERE / "model.pt"))
    if path.exists():
        return path
    repo = os.environ.get("HF_MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename="model.pt"))
    raise FileNotFoundError(
        f"No model at {path}. Set MODEL_PATH, or HF_MODEL_REPO to download model.pt."
    )


torch.set_num_threads(max(1, os.cpu_count() or 1))
MODEL, CFG, THRESHOLDS = load_for_inference(_resolve_model_path(), "cpu")
TRANSFORM = build_transforms(
    CFG["train"]["image_size"], train=False, background=background_value(CFG["preprocess"])
)


def analyse(image: np.ndarray | None):
    if image is None:
        raise gr.Error("Upload a fundus photograph first.")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image[..., :3].astype(np.uint8)

    model_input = preprocess_from_config(image, CFG["preprocess"])
    display = crop_and_resize(image, CFG["preprocess"]["size"])
    x = TRANSFORM(image=model_input)["image"].unsqueeze(0)

    with torch.no_grad():
        score = (MODEL.to_score(MODEL(x)) + MODEL.to_score(MODEL(torch.flip(x, dims=[3])))) / 2
    score = float(score.clamp(0, 4))
    grade = int(scores_to_grades(np.array([score]), THRESHOLDS)[0])
    heat = overlay(display, gradcam(MODEL, x))

    referable = grade >= REFERABLE_GRADE
    rows = "\n".join(
        f"| {g} | {name} |{' **← predicted**' if g == grade else ''} |"
        for g, name in enumerate(GRADE_NAMES)
    )
    summary = (
        f"## Grade {grade}: {GRADE_NAMES[grade]}\n\n"
        f"**Severity score:** {score:.2f} on a 0-4 scale  \n"
        f"**Referable DR:** {'Yes' if referable else 'No'}  \n"
        f"{ADVICE[grade]}\n\n"
        f"| Grade | Meaning | |\n|---|---|---|\n{rows}\n\n"
        f"_Cut points between grades: {', '.join(f'{t:.2f}' for t in THRESHOLDS)}_"
    )
    return heat, model_input, summary


with gr.Blocks(title="Retinal DR Screening") as demo:
    gr.Markdown(
        "# Diabetic Retinopathy Screening\n"
        "Upload a colour fundus photograph. The model predicts the DR grade (0-4) and shows "
        "a Grad-CAM heatmap of the regions that raised the severity score.\n\n" + DISCLAIMER
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Fundus photograph")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            out_text = gr.Markdown()
    with gr.Row():
        out_heat = gr.Image(label="Grad-CAM: regions driving the score")
        out_pre = gr.Image(label="What the model sees (Ben Graham normalised)")
    btn.click(analyse, inputs=inp, outputs=[out_heat, out_pre, out_text])
    gr.Markdown(
        "Trained on Kaggle EyePACS 2015 (pretraining) and APTOS 2019 (fine-tuning); "
        "externally tested on IDRiD."
    )

if __name__ == "__main__":
    demo.launch()
