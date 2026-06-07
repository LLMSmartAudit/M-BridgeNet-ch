"""Pre-check for Direction A (LLM hard-negative mining).

Question: in the HARD region (s1 >= 0.85, where the AP-hurting false positives
live), can the existing features (s1, s2, s3, s5) separate true-bridge pairs
(label=1) from non-bridge pairs (label=0)?  In particular, does the temporal
feature s2 add separation over semantic similarity (s1/s5) alone?

If yes  → the MLP under-uses these features; LLM-mined hard negatives can teach
          it to penalise the long-gap-coincidental FP pattern → Direction A viable.
If no   → the FP/TP ambiguity is irreducible with current features; stop.

Reports 5-fold CV ROC-AUC for feature subsets, with a linear (LogReg) and a
non-linear (GradientBoosting) classifier. NO LLM calls.
"""
from __future__ import annotations
import glob, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DIRS = ["data/cphot/processed/train_all", "data/cphot/processed/test_real"]
S1_MIN = 0.85


def load() -> list[dict]:
    rows = []
    for d in DIRS:
        for f in glob.glob(f"{d}/*.json"):
            if any(s in f for s in ("_s5", "_s6", "_emb")):
                continue
            ev = json.load(open(f))
            eid = ev.get("event_id", "")
            s5 = {}
            p5 = Path(d) / f"{eid}_s5.json"
            if p5.exists():
                s5 = json.loads(p5.read_text())
            for p in ev.get("scored_pairs", []):
                if p.get("label", -1) == -1:
                    continue
                lo, hi = sorted([p["post_a_id"], p["post_b_id"]])
                rows.append({
                    "s1": p["s1"], "s2": p["s2"], "s3": p["s3"],
                    "s5": float(s5.get(f"{lo}||{hi}", 0.0)),
                    "y": int(p["label"]),
                })
    return rows


def cv_auc(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def main():
    rows = load()
    hard = [r for r in rows if r["s1"] >= S1_MIN]
    npos = sum(r["y"] for r in hard); nneg = len(hard) - npos
    print(f"All labeled pairs: {len(rows)}")
    print(f"High-s1 region (s1>={S1_MIN}): {len(hard)}  "
          f"(pos={npos}, neg={nneg}, base-rate={npos/len(hard):.3f})")
    print(f"\n5-fold CV ROC-AUC for separating bridge (1) vs non-bridge (0) "
          f"in the high-s1 region:\n")

    feature_sets = {
        "s1 only":          ["s1"],
        "s5 only (CrossEnc)":["s5"],
        "s1+s5":            ["s1", "s5"],
        "s1+s5+s2 (add temporal)": ["s1", "s5", "s2"],
        "s1+s2+s3+s5 (full)":      ["s1", "s2", "s3", "s5"],
    }
    y = np.array([r["y"] for r in hard])
    print(f"  {'features':28s} {'LogReg AUC':>11} {'GBoost AUC':>11}")
    for name, feats in feature_sets.items():
        X = np.array([[r[f] for f in feats] for r in hard], dtype=float)
        Xs = StandardScaler().fit_transform(X)
        lr = cv_auc(Xs, y, LogisticRegression(max_iter=2000, class_weight="balanced"))
        gb = cv_auc(X, y, GradientBoostingClassifier(random_state=42))
        print(f"  {name:28s} {lr:>11.3f} {gb:>11.3f}")

    print("\nInterpretation:")
    print("  • If 's1+s5+s2' AUC > 's1+s5' AUC by a clear margin → temporal signal")
    print("    is separable in the hard region → hard-negative mining can teach it.")
    print("  • If adding s2/full ~ s5-alone → ambiguity is irreducible; stop.")


if __name__ == "__main__":
    main()
