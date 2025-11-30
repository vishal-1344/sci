Release for GitHub
==================

This folder contains a curated set of scripts, experiment outputs, and paper sources prepared for uploading to GitHub.

Contents
- `sci_clean (3).tex` — paper source (LaTeX)
- `sci (3).pdf` — compiled paper PDF
- `run_sci_signal_v2.py` — MNIST SCI experiment (adaptive controller)
- `run_sci_mitbih_fixed_k.py` — MIT‑BIH fixed-K baseline experiment
- `run_sci_bearings.py` — Bearings SCI experiment
- `run_sci_mnist_robust.py` — MNIST robust experiment script
- `plot_metacognition_hero.py` — plotting helper used in the paper
- `experiments/mitbih_sci_v2/summary.json` — MIT‑BIH summary output
- `experiments/*/per_example.jsonl` — per-example logs for experiments

Notes
- Large/raw datasets (e.g., `mitbih_train.csv`, `mitbih_test.csv`) are not included here. See `FILES_TO_INCLUDE.txt` for paths to copy from the workspace, or download directly from the dataset sources.
- The included `requirements.txt` lists the Python packages required to run the experiments locally.

How to use
1. Run `python prepare_github_release.py` from the repository root to assemble a `repo_ready/` folder containing the selected files.
2. Follow the instructions in `push_instructions.txt` to create a GitHub repository and push the `repo_ready/` folder.

Contact
If you'd like me to initialize the git repo and push (requires your GitHub credentials or GH CLI setup), tell me and I will prepare the commands or attempt the push if you provide the necessary remote URL.
