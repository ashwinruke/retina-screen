"""Make `import retina` work when scripts are run without `pip install -e .`."""
import faulthandler
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# On a hard crash (segmentation fault), print the Python stack of every thread
# instead of dying silently, so we can see which line crashed.
faulthandler.enable(all_threads=True)
