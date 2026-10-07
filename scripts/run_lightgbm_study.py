"""Launch the isolated HGB/LightGBM nested study."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.modeling.lightgbm_comparison.runner import main


if __name__ == "__main__":
    main()
