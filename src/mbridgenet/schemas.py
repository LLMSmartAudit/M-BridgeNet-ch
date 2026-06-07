from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

PLATFORMS = frozenset({"weibo", "zhihu", "bilibili", "douyin"})
PHASES = frozenset({"emergence", "diffusion", "peak", "decline"})


class Route(str, Enum):
    BRIDGE = "bridge"
    MABD = "mabd"
    DISCARD = "discard"
    UNDECIDED = "undecided"


@dataclass
class Post:
    post_id: str
    text: str
    timestamp: datetime
    platform: str
    account_id: str
    event_id: str

    def __post_init__(self):
        if self.platform not in PLATFORMS:
            raise ValueError(
                f"Unknown platform '{self.platform}'. Must be one of {PLATFORMS}"
            )


@dataclass
class CandidatePair:
    post_a: Post          # bridge source (earlier)
    post_b: Post          # bridge target (later)
    s1: float = 0.0       # semantic similarity
    s2: float = 0.0       # lifecycle-aware temporal gap
    s3: float = 0.0       # platform migration rarity
    s4: float = 0.0       # normalized betweenness centrality
    s5: float = 0.0       # CrossEncoder text score (0 if outside top-500 / no s5_map)
    score_S: float = 0.0  # composite score from MLP
    phase: str = ""       # lifecycle phase of post_a
    route: Route = Route.UNDECIDED
    label: int = -1       # 1=bridge, 0=non-bridge, -1=unknown
    score_final: float = 0.0  # after MABD fusion


@dataclass
class DebateRecord:
    is_bridge: bool
    confidence: float
    narrative_relation: Literal["paraphrase", "elaboration", "reaction", "coincidental"]
    deciding_factor: Literal["temporal", "semantic", "rarity", "structural", "none"]
    debate_outcome: Literal["proposer_won", "challenger_won", "balanced"]

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
