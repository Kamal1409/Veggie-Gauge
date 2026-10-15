"""One script to set up and run the whole project.

Works the same on Windows (PowerShell / cmd), macOS and Linux, and only needs a
Python 3.10+ on the PATH to start; everything else is installed into `venv/`.

    python run.py                  # everything: setup, data, train, evaluate, tests
    python run.py setup            # create venv/ and install requirements
    python run.py data [--fresh]   # synthetic dataset, unless real data is present
    python run.py train            # segment every photo, then train the models
    python run.py evaluate [--cnn] # baselines, all metrics and charts
    python run.py test             # the test suite
    python run.py predict PHOTO... # analyse photos -> reports/demo/
    python run.py demo [--port N]  # Gradio demo
    python run.py api [--port N]   # FastAPI service

Add `--cnn` to `all` / `setup` / `evaluate` to include the CNN baseline (installs PyTorch).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DATA = ROOT / "data"
STAMP = VENV / ".requirements-installed"


def say(msg: str) -> None:
    """Print a step heading.

    Args:
        msg: What the step does.
    """
    print(f"\n==> {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> None:
    """Run a command from the project root and stop the script if it fails.

    Args:
        cmd: The command and its arguments.
        **kwargs: Passed on to `subprocess.run`.
    """
    print("   $", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kwargs)
    if result.returncode != 0:
        sys.exit(f"\nFailed (exit code {result.returncode}): {' '.join(str(c) for c in cmd)}")


def py(*args: str) -> None:
    """Run the project's Python (inside venv/) with the given arguments.

    Args:
        *args: Arguments for the interpreter, e.g. `"-m", "eval.evaluate"`.
    """
    if not VENV_PY.exists():
        sys.exit("venv/ not found: run `python run.py setup` first")
    run([str(VENV_PY), *args])


def requirements_hash(cnn: bool) -> str:
    """Fingerprint of the requirement files, so unchanged requirements are not reinstalled.

    Args:
        cnn: Whether the CNN baseline's requirements are included.

    Returns:
        A short hash of the requirement files' contents.
    """
    files = ["requirements.txt"] + (["requirements-baselines.txt"] if cnn else [])
    h = hashlib.sha256()
    for f in files:
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def pip_install(*args: str) -> None:
    """Install packages into venv/, with uv if it is available, otherwise pip.

    Environments created by uv come without pip, so pip is bootstrapped when needed.

    Args:
        *args: Arguments for `pip install`, e.g. `"-r", "requirements.txt"`.
    """
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", str(VENV_PY), *args])
        return
    probe = subprocess.run([str(VENV_PY), "-m", "pip", "--version"], cwd=ROOT, capture_output=True)
    if probe.returncode != 0:
        run([str(VENV_PY), "-m", "ensurepip", "--upgrade"])
    run([str(VENV_PY), "-m", "pip", "install", *args])


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def setup(cnn: bool = False) -> None:
    """Create venv/ if needed and install the requirements (skipped if already up to date).

    Args:
        cnn: Also install PyTorch for the CNN baseline.
    """
    if sys.version_info < (3, 10):  # noqa: UP036  (run.py may be started by an older system Python)
        sys.exit(f"Python 3.10+ is required, this is {sys.version.split()[0]}")
    if not VENV_PY.exists():
        say("Creating virtual environment in venv/")
        run([sys.executable, "-m", "venv", str(VENV)])
    want = requirements_hash(cnn)
    if STAMP.exists() and STAMP.read_text().strip() == want:
        say("Requirements already installed")
        return
    say("Installing requirements")
    pip_install("-r", "requirements.txt")
    if cnn:
        pip_install("-r", "requirements-baselines.txt")
    STAMP.write_text(want)


def data(fresh: bool = False) -> None:
    """Make sure there is a dataset: keep real data, otherwise generate a synthetic one.

    Args:
        fresh: Regenerate the synthetic dataset even if one already exists. Real data
            (a dataset without the SYNTHETIC flag) is never overwritten.
    """
    real = (DATA / "ground_truth.csv").exists() and not (DATA / "SYNTHETIC").exists()
    if real:
        say("Real dataset found in data/: leaving it as it is")
        if not (DATA / "calibration.json").exists():
            sys.exit(
                "data/calibration.json is missing. Calibrate from your ruler photo first, e.g.\n"
                "  venv/Scripts/python -m src.calibrate data/raw/ruler.jpg --points X1 Y1 X2 Y2 --mm 300"
            )
        return
    if (DATA / "ground_truth.csv").exists() and not fresh:
        say("Synthetic dataset already present (use --fresh to regenerate)")
        return
    say("Generating the synthetic dataset (about 4 minutes)")
    py("-m", "tools.synth_data")


def train() -> None:
    """Segment every photo into data/features.csv, then train all the models."""
    say("Segmenting every photo and measuring each item")
    py("-m", "train.build_features", "--overlays")
    say("Training the crop classifier, weight regressors and Small/Big classifiers")
    py("-m", "train.train_models")


def has_torch() -> bool:
    """Whether PyTorch is installed in venv/.

    Returns:
        True if `import torch` works inside the environment.
    """
    return subprocess.run([str(VENV_PY), "-c", "import torch"], cwd=ROOT, capture_output=True).returncode == 0


def evaluate(cnn: bool = False) -> None:
    """Run the baselines, compute every metric and draw the charts.

    Args:
        cnn: Also run the CNN baseline (needs PyTorch; slow on CPU).
    """
    say("Raw-image baseline: HOG + SVM")
    py("-m", "baselines.hog_svm")
    if cnn:
        if has_torch():
            say("Raw-image baseline: CNN")
            py("-m", "baselines.cnn_patch")
        else:
            print("   PyTorch is not installed, skipping the CNN baseline (run `python run.py setup --cnn`)")
    say("Evaluating everything (leave-one-item-out)")
    py("-m", "eval.evaluate")
    say("Drawing charts")
    py("-m", "eval.plots")


def test() -> None:
    """Run the test suite."""
    say("Running the tests")
    py("-m", "pytest", "-q")


def predict(photos: list[str]) -> None:
    """Analyse photos with the trained pipeline, saving JSON and overlays to reports/demo/.

    Args:
        photos: Paths of the photos to analyse.
    """
    say("Analysing photos")
    py("-m", "src.pipeline", *photos, "--out-dir", "reports/demo")


def demo(port: int) -> None:
    """Launch the Gradio demo.

    Args:
        port: Port to serve on.
    """
    say(f"Starting the demo on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    py("-m", "app.gradio_app", "--port", str(port))


def api(port: int) -> None:
    """Launch the FastAPI service.

    Args:
        port: Port to serve on.
    """
    say(f"Starting the API on http://127.0.0.1:{port}  (docs at /docs, Ctrl+C to stop)")
    py("-m", "uvicorn", "app.api:app", "--port", str(port))


def main() -> None:
    """Parse the command line and run the requested steps."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "setup", "data", "train", "evaluate", "test", "predict", "demo", "api"],
    )
    p.add_argument("photos", nargs="*", help="photos for `predict`")
    p.add_argument("--cnn", action="store_true", help="include the CNN baseline (installs PyTorch)")
    p.add_argument("--fresh", action="store_true", help="regenerate the synthetic dataset")
    p.add_argument("--skip-tests", action="store_true", help="`all` without the test suite")
    p.add_argument("--port", type=int, help="port for `demo` (7860) or `api` (8000)")
    args = p.parse_args()

    if args.command == "all":
        setup(args.cnn)
        data(args.fresh)
        train()
        evaluate(args.cnn)
        if not args.skip_tests:
            test()
        say("Done. Results are in reports/ (start with reports/results_table.md). Next:")
        print("   python run.py demo        # try it in the browser")
        print("   python run.py predict data/raw/mixed_0009.jpg")
    elif args.command == "setup":
        setup(args.cnn)
    elif args.command == "data":
        data(args.fresh)
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate(args.cnn)
    elif args.command == "test":
        test()
    elif args.command == "predict":
        if not args.photos:
            sys.exit("usage: python run.py predict PHOTO [PHOTO ...]")
        predict(args.photos)
    elif args.command == "demo":
        demo(args.port or 7860)
    elif args.command == "api":
        api(args.port or 8000)


if __name__ == "__main__":
    main()
