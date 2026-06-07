#!/usr/bin/env python3
"""Download the CPHot dataset + M-BridgeNet checkpoints from the Hugging Face Hub
into the layout the experiment scripts expect (data/cphot/processed/... and
checkpoints/...).

Usage:
    python scripts/download_from_hf.py
    python scripts/download_from_hf.py --repo-id your-org/M-BridgeNet
    python scripts/download_from_hf.py --repo-id your-org/M-BridgeNet --repo-type dataset

Requires: pip install huggingface_hub
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# TODO: replace with the published Hugging Face repo id before release.
DEFAULT_REPO_ID = "TODO/M-BridgeNet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help=f"HF repo id (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"],
                        help="HF repo type (default: dataset)")
    parser.add_argument("--local-dir", default=".",
                        help="Destination root; must be the repo root (default: .)")
    args = parser.parse_args()

    if args.repo_id == DEFAULT_REPO_ID:
        print("ERROR: set --repo-id to the published Hugging Face repo "
              f"(currently the placeholder '{DEFAULT_REPO_ID}').", file=sys.stderr)
        sys.exit(2)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    dest = Path(args.local_dir).resolve()
    print(f"Downloading {args.repo_id} ({args.repo_type}) → {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(dest),
        allow_patterns=["data/**", "checkpoints/**", "DATASET.md"],
    )

    # sanity check
    test_dir = dest / "data" / "cphot" / "processed" / "test_real"
    ckpt = dest / "checkpoints" / "mlp_v25_fold2.pt"
    n_events = len(list(test_dir.glob("*.json"))) if test_dir.exists() else 0
    print(f"\nDone. test_real JSONs: {n_events}   checkpoint present: {ckpt.exists()}")
    if n_events == 0 or not ckpt.exists():
        print("WARNING: expected files missing — check the repo id / layout.",
              file=sys.stderr)
    else:
        print("Ready. Reproduce the headline result with:\n"
              "  MBRIDGENET_NO_FAISS=1 python scripts/evaluate.py \\\n"
              "    --data data/cphot/processed/test_real \\\n"
              "    --checkpoint checkpoints/mlp_v25_fold2.pt --k 5 10 20 50 --low-s2-simonly 0.20")


if __name__ == "__main__":
    main()
