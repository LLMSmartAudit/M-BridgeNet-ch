"""Train the Stage-2 LifecycleMLP with 5-fold cross-validation on CPHot.

Usage:
    python scripts/train.py --data data/cphot/processed \\
                            --output checkpoints/mlp_fold{fold}.pt \\
                            --config configs/default.yaml

Each event JSON must include a "scored_pairs" list of dicts with keys:
    s1, s2, s3, s4, phase (str), label (0 or 1)
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.stage2.scorer import LifecycleMLP, PHASE2IDX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_features(dataset: CPHotDataset, drop_signal: str = "",
                     use_s5: bool = False,
                     data_dir: Path | None = None) -> tuple:
    """Pre-compute (signals, phase_idx, labels) across all events.

    Skips pairs with label == -1 (unknown).
    drop_signal: if "s1", "s2", or "s3", that signal is excluded from features.
    use_s5: if True, appends s5 (CrossEncoder score) from sidecar *_s5.json.
            Pairs without a precomputed s5 get s5=0.0.

    Signal vector per pair: [s1, s2, s3, weibo_frac]       (4 signals, use_s5=False)
                         or [s1, s2, s3, weibo_frac, s5]   (5 signals, use_s5=True)
    """
    import json as _json
    sig_keys = [k for k in ["s1", "s2", "s3"] if k != drop_signal]
    all_signals, all_phases, all_labels = [], [], []
    for event in dataset:
        event_id = event.get("event_id", "")
        # Compute event-level Weibo fraction (same for all pairs in this event)
        posts = event.get("posts", [])
        # posts may be Post objects (CPHotDataset) or dicts (raw JSON)
        n_weibo = sum(
            1 for p in posts
            if (p.platform if hasattr(p, "platform") else p.get("platform")) == "weibo"
        )
        weibo_frac = n_weibo / max(len(posts), 1)

        # Load s5 sidecar if available
        s5_map: dict = {}
        if use_s5 and event_id and data_dir is not None:
            s5_path = Path(data_dir) / f"{event_id}_s5.json"
            if s5_path.exists():
                s5_map = _json.loads(s5_path.read_text())

        for pair in event.get("scored_pairs", []):
            if pair["label"] == -1:
                continue
            base_feats = [pair[k] for k in sig_keys] + [weibo_frac]
            if use_s5:
                pid_a = pair.get("post_a_id", "")
                pid_b = pair.get("post_b_id", "")
                key = "||".join(sorted([pid_a, pid_b]))
                s5 = float(s5_map.get(key, 0.0))
                base_feats = base_feats + [s5]
            all_signals.append(base_feats)
            all_phases.append(PHASE2IDX.get(pair["phase"], 0))
            all_labels.append(float(pair["label"]))

    if not all_signals:
        raise ValueError(
            "No labeled pairs found. Make sure event JSONs contain 'scored_pairs'."
        )

    return (
        torch.tensor(all_signals, dtype=torch.float32),
        torch.tensor(all_phases,  dtype=torch.long),
        torch.tensor(all_labels,  dtype=torch.float32),
    )


def train_fold(
    X_sig: torch.Tensor,
    X_phase: torch.Tensor,
    y: torch.Tensor,
    cfg,
    output_path: Path,
    n_signals: int = 4,
    residual: bool = False,
    alpha: float = 0.5,
) -> float:
    """Train one fold, save checkpoint, return final-epoch mean loss.

    Checkpoint format (new, v24+):
        {"state_dict": OrderedDict, "residual": bool, "alpha": float,
         "n_signals": int}
    Old checkpoints (v23 and earlier) are plain state_dicts for backward
    compatibility; pipeline.py detects the format at load time.
    """
    model = LifecycleMLP(
        n_signals=n_signals,
        hidden=cfg.stage2.mlp_hidden,
        residual=residual,
        alpha=alpha,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.stage2.mlp_lr)
    criterion = nn.BCELoss()
    loader = DataLoader(
        TensorDataset(X_sig, X_phase, y),
        batch_size=cfg.stage2.mlp_batch_size,
        shuffle=True,
    )

    for epoch in range(1, cfg.stage2.mlp_epochs + 1):
        model.train()
        total_loss = 0.0
        for sig, phase, label in loader:
            optimizer.zero_grad()
            pred = model(sig, phase).clamp(1e-7, 1 - 1e-7)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            logger.info("  epoch %d/%d  loss=%.4f",
                        epoch, cfg.stage2.mlp_epochs,
                        total_loss / len(loader))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save with metadata so pipeline can reconstruct the exact architecture
    torch.save(
        {
            "state_dict": model.state_dict(),
            "residual": residual,
            "alpha": alpha,
            "n_signals": n_signals,
        },
        output_path,
    )
    logger.info("  saved → %s  (residual=%s, alpha=%.2f)", output_path, residual, alpha)
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(
        description="Train LifecycleMLP with 5-fold cross-validation"
    )
    parser.add_argument("--data",   required=True,
                        help="Path to directory of CPHot event JSON files")
    parser.add_argument("--output", default="checkpoints/mlp_fold{fold}.pt",
                        help="Output path template (use {fold} placeholder)")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--drop-signal", default="", choices=["", "s1", "s2", "s3"],
                        help="Drop one signal from MLP input (ablation study)")
    parser.add_argument("--residual", action="store_true",
                        help="Use Residual MLP architecture: score = s1 + alpha*correction(phase,s2,s3,weibo_frac). "
                             "Anchors to cosine baseline, prevents s2-collapse for low-temporal-signal events.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Residual correction scale: max correction = ±alpha (default 0.5). "
                             "Only used when --residual is set.")
    parser.add_argument("--use-s5", action="store_true",
                        help="Append s5 (CrossEncoder score) as 5th signal. Requires *_s5.json "
                             "sidecar files in --data directory (run scripts/precompute_s5.py first). "
                             "Pairs without a precomputed s5 receive s5=0.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    dataset = CPHotDataset(Path(args.data))
    use_s5 = getattr(args, "use_s5", False)
    X_sig, X_phase, y = collect_features(dataset, drop_signal=args.drop_signal,
                                          use_s5=use_s5, data_dir=Path(args.data))
    n_signals = 4 - (1 if args.drop_signal else 0) + (1 if use_s5 else 0)
    logger.info("Dataset: %d labeled pairs across %d events, residual=%s, alpha=%.2f",
                len(y), len(dataset), args.residual, args.alpha)

    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sig), start=1):
        logger.info("=== Fold %d/%d  (train=%d, val=%d) ===",
                    fold, cfg.n_folds,
                    len(train_idx), len(val_idx))
        out = Path(args.output.replace("{fold}", str(fold)))
        loss = train_fold(
            X_sig[train_idx], X_phase[train_idx], y[train_idx],
            cfg, out, n_signals=n_signals,
            residual=args.residual, alpha=args.alpha,
        )
        fold_losses.append(loss)

    logger.info("Mean final-epoch loss across %d folds: %.4f",
                cfg.n_folds, float(np.mean(fold_losses)))


if __name__ == "__main__":
    main()
