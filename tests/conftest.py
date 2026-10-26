import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(*args, cwd=ROOT):
    """Run a repo script and fail loudly with its output."""
    proc = subprocess.run(
        [sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise AssertionError(f"{args} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    run(ROOT / "scripts/make_synthetic.py", "--out", out)
    return out


@pytest.fixture(scope="session")
def manifests_dir(synthetic, tmp_path_factory):
    out = tmp_path_factory.mktemp("manifests")
    run(
        ROOT / "scripts/prepare_data.py",
        "--aptos", synthetic / "aptos",
        "--eyepacs", synthetic / "eyepacs",
        "--idrid", synthetic / "idrid",
        "--out-dir", out,
    )
    return out
