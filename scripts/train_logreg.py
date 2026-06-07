"""Train and save a Logistic Regression baseline on (s1, s2, s3) features.

LR-NoPhase: flat logistic regression WITHOUT lifecycle phase conditioning.
Used as an ablation baseline in the paper (Table 1).

Usage:
    python scripts/train_logreg.py \
        --train data/cphot/processed/train_all \
        --output checkpoints/logreg_nophase.pkl
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_train_data(train_dir: Path):
    X, y = [], []
    n_events = 0
    for path in sorted(train_dir.glob("*.json")):
        if "_emb" in path.stem:
            continue
        with open(path) as f:
            event = json.load(f)
        pairs = event.get("scored_pairs", [])
        for p in pairs:
            if p.get("label", -1) == -1:
                continue
            X.append([p["s1"], p["s2"], p["s3"]])
            y.append(int(p["label"]))
        n_events += 1
    logger.info("Loaded %d events, %d pairs (%d bridges, %d non-bridges)",
                n_events, len(y), sum(y), len(y) - sum(y))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/cphot/processed/train_all")
    parser.add_argument("--output", default="checkpoints/logreg_nophase.pkl")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train = load_train_data(Path(args.train))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed,
                             class_weight="balanced", solver="lbfgs")
    clf.fit(X_scaled, y_train)

    train_acc = clf.score(X_scaled, y_train)
    coef_dict = dict(zip(["s1", "s2", "s3"], clf.coef_[0].round(4)))
    logger.info("LR train accuracy: %.4f  coefs: %s  intercept: %.4f",
                train_acc, coef_dict, float(clf.intercept_[0]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    logger.info("Saved LR model to %s", out_path)
    print(f"LR coefs: {coef_dict}")
    print(f"Intercept: {clf.intercept_[0]:.4f}")


if __name__ == "__main__":
    main()
