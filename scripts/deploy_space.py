"""Publish the Gradio demo to a Hugging Face Space in one command.

Needs a Hugging Face access token with *write* permission in the HF_TOKEN
environment variable. On Kaggle, store it under Add-ons -> Secrets and load it:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

Then:

    python scripts/deploy_space.py --model outputs/model.pt --space <hf-username>/retina-dr-screening

Re-running updates the same Space (new model, new code).
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_folder(model: Path, dest: Path, github_url: str | None) -> None:
    """Lay out what the Space needs: app.py, requirements.txt, README.md
    (Space settings), the retina package and model.pt."""
    shutil.copy(ROOT / "app" / "app.py", dest / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", dest / "requirements.txt")
    readme = (ROOT / "app" / "SPACE_README.md").read_text()
    if github_url:
        readme = readme.replace("https://github.com/ashwinruke/retina-screen", github_url)
    (dest / "README.md").write_text(readme)
    shutil.copytree(ROOT / "src" / "retina", dest / "retina",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(model, dest / "model.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.pt exported by evaluate.py --export")
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--private", action="store_true", help="Create the Space as private")
    ap.add_argument("--github-url", help="Repo link shown on the Space page")
    ap.add_argument("--dry-run", action="store_true", help="Build the folder only; print where it is")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    tmp = Path(tempfile.mkdtemp(prefix="space_"))
    build_folder(model, tmp, args.github_url)
    if args.dry_run:
        print("Space folder built at", tmp)
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                print(" ", p.relative_to(tmp))
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access (see the docstring).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(tmp), repo_id=args.space, repo_type="space",
                      commit_message="Deploy DR screening demo")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")
    print("The first build takes a few minutes; watch the 'Logs' tab on the Space page.")


if __name__ == "__main__":
    main()
