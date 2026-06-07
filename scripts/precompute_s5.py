"""Precompute CrossEncoder (s5) scores for scored_pairs in training events
and for top-K Stage-1 candidates in test events.

Saves sidecar files:
    <event_dir>/<event_id>_s5.json  →  {"pair_id": score, ...}

pair_id = "{post_a}||{post_b}" (alphabetically sorted for canonical lookup)

Usage:
    # Precompute for training events (scored_pairs)
    MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/precompute_s5.py \\
        --data data/cphot/processed/train_all \\
        --mode train \\
        --ckpt checkpoints/crossencoder_fold5

    # Precompute for test events (top-500 Stage-1 candidates)
    MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/precompute_s5.py \\
        --data data/cphot/processed/test_real \\
        --mode test \\
        --ckpt checkpoints/crossencoder_fold5 \\
        --top-k 500
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CrossEncoder model (mirrors evaluate_crossencoder.py) ─────────────────────
class CrossEncoder(nn.Module):
    """Mirrors train_crossencoder.py: BERT + Dropout(0.1) + Linear(768→1)."""
    def __init__(self, bert_model):
        super().__init__()
        self.bert    = bert_model
        hidden       = bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head    = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls)).squeeze(-1)


def load_crossencoder(ckpt_dir: Path, device: torch.device):
    bert = AutoModel.from_pretrained(str(ckpt_dir))
    tok  = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = CrossEncoder(bert).to(device)
    head_path = ckpt_dir / "head.pt"
    if head_path.exists():
        # head.pt is a dict {"head": OrderedDict, "dropout": OrderedDict}
        raw = torch.load(str(head_path), map_location=device, weights_only=False)
        if isinstance(raw, dict) and "head" in raw:
            model.head.load_state_dict(raw["head"])
            if "dropout" in raw:
                model.dropout.load_state_dict(raw["dropout"])
        else:
            model.head.load_state_dict(raw)
    model.eval()
    logger.info("Loaded CrossEncoder from %s", ckpt_dir)
    return model, tok


def score_pairs_batch(
    model, tokenizer, device,
    text_pairs: list[tuple[str, str]],
    batch_size: int = 64,
    max_length: int = 256,
) -> list[float]:
    scores = []
    for i in range(0, len(text_pairs), batch_size):
        batch = text_pairs[i: i + batch_size]
        enc = tokenizer(
            [a for a, b in batch], [b for a, b in batch],
            return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc)
            probs  = torch.sigmoid(logits).cpu().numpy()
        scores.extend(probs.tolist())
    return scores


def pair_key(a: str, b: str) -> str:
    """Canonical (sorted) pair identifier."""
    lo, hi = (a, b) if a < b else (b, a)
    return f"{lo}||{hi}"


# ── train mode: score annotated scored_pairs ──────────────────────────────────
def process_train_event(
    event_path: Path, model, tokenizer, device, out_dir: Path
) -> None:
    with open(event_path) as f:
        d = json.load(f)

    event_id = d["event_id"]
    out_file = out_dir / f"{event_id}_s5.json"
    if out_file.exists():
        logger.info("  %s: s5 already computed, skipping", event_id)
        return

    scored_pairs = d.get("scored_pairs", [])
    if not scored_pairs:
        logger.warning("  %s: no scored_pairs, skipping", event_id)
        return

    # Build text map
    text_map = {p["post_id"]: p.get("text", "") for p in d["posts"]}

    text_pairs  = []
    pair_ids    = []
    for sp in scored_pairs:
        pid_a = sp.get("post_a_id") or sp.get("source_id") or ""
        pid_b = sp.get("post_b_id") or sp.get("target_id") or ""
        if not pid_a or not pid_b:
            continue
        ta = text_map.get(pid_a, "")
        tb = text_map.get(pid_b, "")
        if not ta or not tb:
            continue
        pair_ids.append(pair_key(pid_a, pid_b))
        text_pairs.append((ta, tb))

    if not text_pairs:
        logger.warning("  %s: no valid text pairs, skipping", event_id)
        return

    scores = score_pairs_batch(model, tokenizer, device, text_pairs)
    s5_map = {pid: float(score) for pid, score in zip(pair_ids, scores)}

    with open(out_file, "w") as f:
        json.dump(s5_map, f)
    logger.info("  %s: scored %d pairs → %s", event_id, len(s5_map), out_file.name)


# ── test mode: score top-K Stage-1 candidates ─────────────────────────────────
def process_test_event(
    event_path: Path, model, tokenizer, device, out_dir: Path,
    top_k: int = 500, tau_coarse: float = 0.50,
) -> None:
    with open(event_path) as f:
        d = json.load(f)

    event_id = d["event_id"]
    out_file = out_dir / f"{event_id}_s5.json"
    if out_file.exists():
        logger.info("  %s: s5 already computed, skipping", event_id)
        return

    posts_raw = d["posts"]
    text_map = {p["post_id"]: p.get("text", "") for p in posts_raw}

    # Load precomputed BGE embeddings from npz sidecar
    emb_path = event_path.parent / f"{event_id}_emb.npz"
    if not emb_path.exists():
        logger.warning("  %s: no _emb.npz sidecar, skipping", event_id)
        return

    emb_data = np.load(emb_path)
    posts_ids = [p["post_id"] for p in posts_raw]
    platforms = {p["post_id"]: p.get("platform", "?") for p in posts_raw}

    # Build per-platform embedding matrices
    from collections import defaultdict
    plat_posts = defaultdict(list)
    for pid in posts_ids:
        if pid in emb_data:
            plat_posts[platforms[pid]].append(pid)

    plat_keys = list(plat_posts.keys())
    all_pairs = []
    for i in range(len(plat_keys)):
        for j in range(i + 1, len(plat_keys)):
            pa, pb = plat_keys[i], plat_keys[j]
            ids_a = plat_posts[pa]
            ids_b = plat_posts[pb]
            a_embs = np.stack([emb_data[pid] for pid in ids_a])
            b_embs = np.stack([emb_data[pid] for pid in ids_b])
            a_norm = a_embs / (np.linalg.norm(a_embs, axis=1, keepdims=True) + 1e-8)
            b_norm = b_embs / (np.linalg.norm(b_embs, axis=1, keepdims=True) + 1e-8)
            cos_mat = a_norm @ b_norm.T
            idxs = np.argwhere(cos_mat >= tau_coarse)
            for r, c in idxs:
                all_pairs.append((float(cos_mat[r, c]), ids_a[r], ids_b[c]))

    all_pairs.sort(key=lambda x: -x[0])
    top_pairs = all_pairs[:top_k]

    if not top_pairs:
        logger.warning("  %s: no Stage-1 candidates, skipping", event_id)
        return

    text_pairs = []
    pair_ids   = []
    for _, pid_a, pid_b in top_pairs:
        ta = text_map.get(pid_a, "")
        tb = text_map.get(pid_b, "")
        if ta and tb:
            pair_ids.append(pair_key(pid_a, pid_b))
            text_pairs.append((ta, tb))

    scores = score_pairs_batch(model, tokenizer, device, text_pairs)
    s5_map = {pid: float(score) for pid, score in zip(pair_ids, scores)}

    with open(out_file, "w") as f:
        json.dump(s5_map, f)
    logger.info("  %s: scored %d top-%d candidates → %s",
                event_id, len(s5_map), top_k, out_file.name)


def main():
    parser = argparse.ArgumentParser(description="Precompute CrossEncoder s5 scores")
    parser.add_argument("--data", required=True, help="Event directory (train_all or test_real)")
    parser.add_argument("--ckpt", required=True, help="CrossEncoder checkpoint directory")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--top-k", type=int, default=500,
                        help="(test mode) number of top Stage-1 candidates to score")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data)
    ckpt_dir = Path(args.ckpt)
    device   = torch.device(args.device)

    model, tokenizer = load_crossencoder(ckpt_dir, device)

    json_files = sorted(data_dir.glob("*.json"))
    # Filter out _s5.json files
    json_files = [f for f in json_files if not f.name.endswith("_s5.json")]
    logger.info("Processing %d events in %s mode ...", len(json_files), args.mode)

    for i, jf in enumerate(json_files, 1):
        logger.info("[%d/%d] %s", i, len(json_files), jf.stem)
        if args.mode == "train":
            process_train_event(jf, model, tokenizer, device, data_dir)
        else:
            process_test_event(jf, model, tokenizer, device, data_dir, top_k=args.top_k)

    logger.info("Done. s5 files written to %s", data_dir)


if __name__ == "__main__":
    main()
