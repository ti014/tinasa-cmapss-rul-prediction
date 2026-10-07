"""Run independent dataset studies in separate processes, four fit threads each."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from cmapss.project import FOLLOWUP_RUN


def run_one(dataset: str, root: Path) -> tuple[str, dict]:
    """Keep each study log and process isolated; resume only hash-matching fits."""
    logs = root / "tmp"
    with (logs / f"followup-{dataset}.log").open("w", encoding="utf-8") as stdout, (logs / f"followup-{dataset}.err").open("w", encoding="utf-8") as stderr:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", dataset], cwd=root,
                       stdout=stdout, stderr=stderr, check=True)
    info = json.loads((root / FOLLOWUP_RUN / dataset / "run_info.json").read_text(encoding="utf-8"))
    return dataset, info


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        from cmapss.modeling.followup import StudyConfig, run_study
        run_study(sys.argv[2], root, root / FOLLOWUP_RUN / sys.argv[2], StudyConfig())
    else:
        (root / "tmp").mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_one, dataset, root) for dataset in ("FD001", "FD002", "FD004")]
            results = dict(future.result() for future in futures)
        (root / FOLLOWUP_RUN / "completion.json").write_text(
            json.dumps({"status": "complete", "datasets": results}, indent=2), encoding="utf-8")
        print("All three studies complete.", flush=True)
