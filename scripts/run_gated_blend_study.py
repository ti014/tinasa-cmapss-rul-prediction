"""Launch the isolated prediction-derived HGB gating study.

Add the local source tree so this entrypoint works on Windows and Linux.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.modeling.gated_blend.runner import main

if __name__ == "__main__":
    main()
