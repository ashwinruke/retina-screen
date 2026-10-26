#!/usr/bin/env bash
# Assemble a folder you can push to a Hugging Face Space.
#
#   bash scripts/build_space.sh outputs/model.pt space/
#   cd space && git init && git remote add origin https://huggingface.co/spaces/<user>/retina-dr
#   git lfs install && git lfs track "*.pt" && git add . && git commit -m init && git push -u origin main
#
# For a model over ~100 MB, upload it to a HF *model* repo instead, skip the
# copy below, and set HF_MODEL_REPO=<user>/<repo> in the Space's settings.
set -euo pipefail
MODEL=${1:?usage: build_space.sh MODEL_PT OUT_DIR}
OUT=${2:?usage: build_space.sh MODEL_PT OUT_DIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$OUT"
cp "$ROOT/app/app.py" "$ROOT/app/requirements.txt" "$OUT/"
cp "$ROOT/app/SPACE_README.md" "$OUT/README.md"
rm -rf "$OUT/retina" && cp -r "$ROOT/src/retina" "$OUT/retina"
find "$OUT/retina" -name "__pycache__" -prune -exec rm -rf {} +
cp "$MODEL" "$OUT/model.pt"
echo "Space folder ready: $OUT"
