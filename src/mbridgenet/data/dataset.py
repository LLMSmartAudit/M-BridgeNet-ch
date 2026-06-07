from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mbridgenet.schemas import Post


def load_event(path: Path) -> dict[str, Any]:
    """Load a single CPHot event JSON file.

    Returns:
        {
          "event_id": str,
          "posts": List[Post],
          "bridge_pairs": List[Tuple[str, str]],   # (post_id_a, post_id_b)
          "hourly_volumes": List[int],
        }
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    event_id = raw["event_id"]
    posts = [
        Post(
            post_id=p["post_id"],
            text=p["text"],
            timestamp=datetime.fromisoformat(p["timestamp"]),
            platform=p["platform"],
            account_id=p["account_id"],
            event_id=event_id,
        )
        for p in raw["posts"]
    ]
    bridge_pairs = [tuple(bp) for bp in raw.get("bridge_pairs", [])]
    hourly_volumes = raw.get("hourly_volumes", [])
    scored_pairs = raw.get("scored_pairs", [])

    return {
        "event_id": event_id,
        "posts": posts,
        "bridge_pairs": bridge_pairs,
        "hourly_volumes": hourly_volumes,
        "scored_pairs": scored_pairs,
    }


class CPHotDataset:
    """Lazy-loading dataset over a directory of CPHot event JSON files."""

    def __init__(self, root: Path):
        self._files = sorted(
            f for f in Path(root).glob("*.json")
            if not (f.name.endswith("_s5.json") or f.name.endswith("_s6.json"))
        )
        if not self._files:
            raise FileNotFoundError(f"No JSON files found under {root}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return load_event(self._files[idx])

    def __iter__(self):
        for f in self._files:
            yield load_event(f)
