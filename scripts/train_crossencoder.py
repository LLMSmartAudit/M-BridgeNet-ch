"""Train a MacBERT cross-encoder for bridge-pair classification.

Architecture:
    [CLS] post_a_text [SEP] post_b_text [SEP]
    → hfl/chinese-macbert-base (102M params)
    → Dropout(0.1) → Linear(768 → 1) → sigmoid
    Loss: BCEWithLogitsLoss with pos_weight = n_neg / n_pos
    Optim: AdamW lr=2e-5, weight_decay=0.01
    Epochs: 3, early stopping on val loss (patience=1)

Only processes events that have both:
    - a raw JSON in data/cphot/raw/  (for post text)
    - a processed JSON in data/cphot/processed/train_all/  (for labels)

Usage:
    .venv/bin/python scripts/train_crossencoder.py
    .venv/bin/python scripts/train_crossencoder.py \\
        --train-dir data/cphot/processed/train_all \\
        --raw-dir   data/cphot/raw \\
        --output    "checkpoints/crossencoder_fold{fold}" \\
        --epochs 3 --batch-size 16 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise SystemExit("Run: pip install torch transformers")


# ── Dataset ──────────────────────────────────────────────────────────────────

class BridgePairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str, int]],
        tokenizer,
        max_length: int = 512,
    ):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.pairs[idx]
        # Symmetric truncation: 256 tokens per side (accounting for special tokens)
        enc = self.tokenizer(
            text_a,
            text_b,
            truncation="longest_first",
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(label, dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class CrossEncoder(nn.Module):
    def __init__(self, model_name: str = "hfl/chinese-macbert-base"):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head    = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return self.head(self.dropout(cls)).squeeze(-1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pairs(
    train_dir: Path,
    raw_dir: Path,
) -> dict[str, list[tuple[str, str, int]]]:
    """Return {event_id: [(text_a, text_b, label), ...]} for events with raw text."""
    event_pairs: dict[str, list] = {}

    for proc_file in sorted(train_dir.glob("*.json")):
        event_id = proc_file.stem
        raw_file = raw_dir / f"{event_id}.json"
        # Handle the case where raw_file is a directory containing the actual JSON
        if raw_file.is_dir():
            inner = raw_file / f"{event_id}.json"
            if inner.exists():
                raw_file = inner
            else:
                logger.debug("Skipping %s — raw JSON dir has no inner file", event_id)
                continue
        elif not raw_file.exists():
            logger.debug("Skipping %s — no raw JSON", event_id)
            continue

        with open(raw_file, encoding="utf-8") as f:
            raw = json.load(f)
        with open(proc_file, encoding="utf-8") as f:
            proc = json.load(f)

        # build post_id → text map from raw
        text_map = {p["post_id"]: p.get("text", "") for p in raw["posts"]}

        pairs: list[tuple[str, str, int]] = []
        for sp in proc["scored_pairs"]:
            if sp["label"] == -1:
                continue
            ta = text_map.get(sp["post_a_id"], "")
            tb = text_map.get(sp["post_b_id"], "")
            if not ta or not tb:
                continue
            pairs.append((ta, tb, int(sp["label"])))

        if pairs:
            event_pairs[event_id] = pairs
            pos = sum(1 for _, _, l in pairs if l == 1)
            logger.info("  %s: %d pairs (%d pos, %d neg)",
                        event_id, len(pairs), pos, len(pairs) - pos)

    return event_pairs


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(
    train_pairs: list[tuple[str, str, int]],
    val_pairs:   list[tuple[str, str, int]],
    model_name:  str,
    output_dir:  Path,
    epochs:      int = 3,
    batch_size:  int = 16,
    lr:          float = 2e-5,
    weight_decay: float = 0.01,
    seed:        int = 42,
    max_length:  int = 512,
    force_device: str | None = None,
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    if force_device:
        device = torch.device(force_device)
    else:
        device = (
            torch.device("mps")  if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
    logger.info("  device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = CrossEncoder(model_name).to(device)

    n_pos = sum(1 for _, _, l in train_pairs if l == 1)
    n_neg = sum(1 for _, _, l in train_pairs if l == 0)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_ds = BridgePairDataset(train_pairs, tokenizer, max_length=max_length)
    val_ds   = BridgePairDataset(val_pairs,   tokenizer, max_length=max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(len(train_dl), 1)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                logits = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    token_type_ids = batch["token_type_ids"].to(device),
                )
                val_loss += criterion(logits, batch["label"].to(device)).item()
        avg_val_loss = val_loss / max(len(val_dl), 1)

        logger.info("  epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, epochs, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= 1:   # patience=1
                logger.info("  early stopping at epoch %d", epoch)
                break

    # Save best checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    head_path = output_dir / "head.pt"
    torch.save({"head": model.head.state_dict(), "dropout": model.dropout.state_dict()},
               head_path)
    logger.info("  saved → %s  (val_loss=%.4f)", output_dir, best_val_loss)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MacBERT CrossEncoder for bridge detection")
    parser.add_argument("--train-dir",   default="data/cphot/processed/train_all")
    parser.add_argument("--raw-dir",     default="data/cphot/raw")
    parser.add_argument("--output",      default="checkpoints/crossencoder_fold{fold}",
                        help="Output dir pattern; {fold} is replaced with fold number 1–5")
    parser.add_argument("--model",       default="hfl/chinese-macbert-base")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--max-length",  type=int, default=512,
                        help="Max token length per pair (default 512; use 128 for speed)")
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--folds",       type=int, default=5)
    parser.add_argument("--fold-only",   type=int, default=None,
                        help="Train only this specific fold (1-indexed); skips all others")
    parser.add_argument("--device",      default=None,
                        help="Force device: cpu | mps | cuda (default: auto-detect)")
    args = parser.parse_args()

    random.seed(args.seed)

    train_dir = Path(args.train_dir)
    raw_dir   = Path(args.raw_dir)

    logger.info("Loading pairs from %s × %s ...", train_dir, raw_dir)
    event_pairs = load_pairs(train_dir, raw_dir)
    event_ids   = sorted(event_pairs.keys())
    logger.info("Total events with text: %d", len(event_ids))
    total_pairs = sum(len(v) for v in event_pairs.values())
    logger.info("Total labeled pairs: %d", total_pairs)

    if len(event_ids) < args.folds:
        raise SystemExit(
            f"Need ≥{args.folds} events for {args.folds}-fold CV; got {len(event_ids)}"
        )

    # Event-level k-fold splits
    random.shuffle(event_ids)
    fold_size = len(event_ids) // args.folds

    for fold in range(1, args.folds + 1):
        if args.fold_only is not None and fold != args.fold_only:
            logger.info("Skipping fold %d (--fold-only %d)", fold, args.fold_only)
            continue

        val_start  = (fold - 1) * fold_size
        val_end    = fold * fold_size if fold < args.folds else len(event_ids)
        val_ids    = event_ids[val_start:val_end]
        train_ids  = [e for e in event_ids if e not in set(val_ids)]

        train_pairs: list[tuple[str, str, int]] = []
        for eid in train_ids:
            train_pairs.extend(event_pairs[eid])

        val_pairs: list[tuple[str, str, int]] = []
        for eid in val_ids:
            val_pairs.extend(event_pairs[eid])

        n_pos = sum(1 for _, _, l in train_pairs if l == 1)
        logger.info(
            "\n=== Fold %d/%d  (train=%d pairs / %d events,  val=%d pairs / %d events) ===",
            fold, args.folds, len(train_pairs), len(train_ids), len(val_pairs), len(val_ids),
        )
        logger.info("  pos/neg in train: %d / %d", n_pos, len(train_pairs) - n_pos)

        output_dir = Path(args.output.format(fold=fold))
        train_fold(
            train_pairs  = train_pairs,
            val_pairs    = val_pairs,
            model_name   = args.model,
            output_dir   = output_dir,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            seed         = args.seed + fold,
            max_length   = args.max_length,
            force_device = args.device,
        )

    logger.info("\nAll %d folds complete.", args.folds)


if __name__ == "__main__":
    main()
