"""Split processed CPHot events into train and test directories.

Usage:
    python scripts/split_events.py \\
        --input  data/cphot/processed \\
        --output data/cphot \\
        --test-ratio 0.2 \\
        --seed 42
"""
from __future__ import annotations
import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Directory containing processed event JSON files")
    parser.add_argument("--output",     required=True,
                        help="Parent output directory; train/ and test/ created inside")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of events for test set (default: 0.2)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input)
    events = sorted(in_dir.glob("*.json"))
    if not events:
        logger.error("No JSON files found in %s", in_dir)
        return

    random.seed(args.seed)
    random.shuffle(events)
    n_test = max(1, int(len(events) * args.test_ratio))
    test_files  = events[:n_test]
    train_files = events[n_test:]

    out_dir   = Path(args.output)
    train_dir = out_dir / "train"
    test_dir  = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy(f, train_dir / f.name)
    for f in test_files:
        shutil.copy(f, test_dir / f.name)

    logger.info("Split: %d train  /  %d test", len(train_files), len(test_files))
    logger.info("Train → %s", train_dir)
    logger.info("Test  → %s", test_dir)


if __name__ == "__main__":
    main()
