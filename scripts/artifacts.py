"""Save a run's results to a private Hugging Face repo, and restore them later.

Kaggle wipes /kaggle/working when a session ends. Push right after training,
and pull at the start of the next session instead of retraining:

    python scripts/artifacts.py push --tag mvp --run baseline_aptos
    python scripts/artifacts.py pull --tag mvp

Needs HF_TOKEN (write access) in the environment; on Kaggle:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

What is saved under <tag>/ in the repo:
    model.pt                  outputs/model.pt (weights + config + cut points)
    runs/<run>/               best.pt, history.csv, train.log, config.yaml
    eval/                     outputs/eval/*.json and *_predictions.csv
    manifests/                data/manifests/*.csv (the exact splits used)

The repo (default <your-hf-username>/retina-screen-runs) is created private.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ["best.pt", "history.csv", "train.log", "config.yaml"]


def stage(tag: str, runs: list[str], include_last: bool, dest: Path, skip_model: bool = False) -> list[Path]:
    """Copy what should be saved into dest/<tag>/; return the files copied."""
    base = dest / tag
    copied = []

    def put(src: Path, rel: str) -> None:
        if src.exists():
            target = base / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, target)
            copied.append(target)

    if not skip_model:
        put(ROOT / "outputs" / "model.pt", "model.pt")
    for run in runs:
        run_dir = ROOT / "outputs" / run
        if not run_dir.is_dir():
            raise SystemExit(f"No run folder {run_dir}")
        for name in RUN_FILES + (["last.pt"] if include_last else []):
            put(run_dir / name, f"runs/{run}/{name}")
    for p in sorted((ROOT / "outputs" / "eval").glob("*")):
        if p.suffix in {".json", ".csv"}:
            put(p, f"eval/{p.name}")
    for p in sorted((ROOT / "data" / "manifests").glob("*.csv")):
        put(p, f"manifests/{p.name}")
    if not copied:
        raise SystemExit("Nothing to save: no outputs/model.pt, run folders, eval reports or manifests found.")
    return copied


def restore(src: Path, tag: str) -> list[str]:
    """Copy a downloaded <tag>/ folder back into outputs/ and data/manifests/."""
    base = src / tag
    if not base.is_dir():
        raise SystemExit(f"Tag '{tag}' not found in the repo.")
    restored = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if rel.parts[0] == "model.pt":
            target = ROOT / "outputs" / "model.pt"
        elif rel.parts[0] == "runs":
            target = ROOT / "outputs" / Path(*rel.parts[1:])
        elif rel.parts[0] == "eval":
            target = ROOT / "outputs" / "eval" / Path(*rel.parts[1:])
        elif rel.parts[0] == "manifests":
            target = ROOT / "data" / "manifests" / Path(*rel.parts[1:])
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, target)
        restored.append(str(target.relative_to(ROOT)))
    return restored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull", "list"])
    ap.add_argument("--tag", help="Name for this set of results, e.g. mvp, eyepacs-pretrain")
    ap.add_argument("--run", nargs="*", default=[], help="Run folders under outputs/ to save (push)")
    ap.add_argument("--include-last", action="store_true", help="Also save last.pt (needed to resume training)")
    ap.add_argument("--skip-model", action="store_true",
                    help="Don't save outputs/model.pt (e.g. mid-pipeline, when it belongs to an older run)")
    ap.add_argument("--repo", help="Hugging Face repo id (default <user>/retina-screen-runs)")
    ap.add_argument("--dry-run", action="store_true", help="push: stage files locally and list them")
    args = ap.parse_args()

    if args.action in ("push", "pull") and not args.tag:
        ap.error("--tag is required")

    if args.action == "push" and args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="artifacts_"))
        for p in stage(args.tag, args.run, args.include_last, tmp, args.skip_model):
            print(f"  {p.relative_to(tmp)}  ({p.stat().st_size / 2**20:.1f} MB)")
        shutil.rmtree(tmp)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access (see the docstring).")
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    repo = args.repo or f"{api.whoami()['name']}/retina-screen-runs"

    if args.action == "list":
        tags = sorted({f.split("/")[0] for f in api.list_repo_files(repo) if "/" in f})
        print("Saved tags:", ", ".join(tags) or "(none)")
        return

    if args.action == "push":
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="artifacts_"))
        files = stage(args.tag, args.run, args.include_last, tmp, args.skip_model)
        api.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="model",
                          commit_message=f"Save results: {args.tag}")
        shutil.rmtree(tmp)
        print(f"Saved {len(files)} files to https://huggingface.co/{repo}/tree/main/{args.tag}")
        return

    local = Path(snapshot_download(repo, repo_type="model", token=token, allow_patterns=[f"{args.tag}/*"]))
    for path in restore(local, args.tag):
        print("restored", path)


if __name__ == "__main__":
    main()
