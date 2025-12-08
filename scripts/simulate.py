from pathlib import Path
import sys

# --- Make sure Python can see the `src` folder ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# --- Now normal imports work ---
from satnet.simulation.engine import run_simulation


if __name__ == "__main__":
    run_simulation()
