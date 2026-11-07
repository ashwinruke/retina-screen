"""Publish the demo to a Hugging Face Space in one command.

Default: a free *static* Space. The page in web/ runs the ONNX model in the
visitor's browser, so no server (and no paid plan) is needed. Export the
model first:

    python scripts/export_onnx.py --model outputs/model.pt --out-dir web \
        --check-images cache/aptos.csv --report outputs/eval/mvp_aptos_test.json

Then, with a Hugging Face token that has *write* access in HF_TOKEN:

    python scripts/deploy_space.py --space <hf-username>/retina-dr-screening

On Kaggle, store the token under Add-ons -> Secrets and load it first:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

--gradio deploys the server-side Gradio app (app/) instead. Hugging Face now
requires a PRO subscription for Gradio Spaces on CPU.

Re-running updates the same Space.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/ashwinruke/retina-screen"


def _readme(name: str, github_url: str | None) -> str:
    text = (ROOT / "app" / name).read_text()
    return text.replace(GITHUB, github_url) if github_url else text


def build_static(site_dir: Path, dest: Path, github_url: str | None) -> None:
    for name in ("index.html", "app.js", "model.onnx", "model.json"):
        src = site_dir / name
        if not src.exists():
            hint = " Run scripts/export_onnx.py first." if name.startswith("model") else ""
            raise SystemExit(f"Missing {src}.{hint}")
        shutil.copy(src, dest / name)
    (dest / "README.md").write_text(_readme("STATIC_SPACE_README.md", github_url))


def build_gradio(model: Path, dest: Path, github_url: str | None) -> None:
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")
    shutil.copy(ROOT / "app" / "app.py", dest / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", dest / "requirements.txt")
    (dest / "README.md").write_text(_readme("SPACE_README.md", github_url))
    shutil.copytree(ROOT / "src" / "retina", dest / "retina",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(model, dest / "model.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--site-dir", default=str(ROOT / "web"), help="Folder with index.html, app.js and the export")
    ap.add_argument("--gradio", action="store_true", help="Deploy the Gradio app instead (needs HF PRO)")
    ap.add_argument("--model", default="outputs/model.pt", help="model.pt, for --gradio only")
    ap.add_argument("--private", action="store_true", help="Create the Space as private")
    ap.add_argument("--github-url", help="Repo link shown on the Space page")
    ap.add_argument("--dry-run", action="store_true", help="Build the folder only; print where it is")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="space_"))
    if args.gradio:
        build_gradio(Path(args.model), tmp, args.github_url)
    else:
        build_static(Path(args.site_dir), tmp, args.github_url)

    if args.dry_run:
        print("Space folder built at", tmp)
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(tmp)}  ({p.stat().st_size / 2**20:.1f} MB)")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access (see the docstring).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio" if args.gradio else "static",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(tmp), repo_id=args.space, repo_type="space",
                      commit_message="Deploy DR screening demo")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")
    print("Static Spaces are live within a minute or so; Gradio Spaces take a few minutes to build.")


if __name__ == "__main__":
    main()
