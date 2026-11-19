# Experiment archive status

The full V11–V16 experiment scripts and measurement logs have not yet been pushed to GitHub. The local `swp-lm.git.bundle` and `swp-lm-source.zip` retain the complete 132-file archive.

To restore raw experiment ZIPs locally, use `python scripts/restore_experiment_archives.py --input-dir PATH`. This repository intentionally excludes teacher weights, trained checkpoints and generated training corpora. The minimal core in `src/swp_lm/` and `tests/` is executable independently of those assets.
