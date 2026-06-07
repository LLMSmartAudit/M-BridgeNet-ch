from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent  # repo root


@dataclass
class Stage1Config:
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    tau_coarse: float = 0.50
    max_time_gap_hours: float = 72.0
    adaptive_time_gap: bool = False  # compute gap from IQR of post timestamps
    top_k_candidates: int = 5000
    faiss_ef_search: int = 64
    faiss_m: int = 32


@dataclass
class Stage2Config:
    tau_fine: float = 0.65
    tau_high: float = 0.72
    tau_low: float = 0.55
    phase_windows: Dict[str, int] = field(default_factory=lambda: {
        "emergence": 6, "diffusion": 24, "peak": 48, "decline": 72
    })
    lifecycle_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "emergence_start": 0.10,
        "diffusion_start": 0.50,
        "peak_start": 0.80,
    })
    smoothing_window: int = 6
    mlp_hidden: int = 64
    mlp_lr: float = 1e-3
    mlp_epochs: int = 50
    mlp_batch_size: int = 256
    # Per-bucket MABD promotion: top-K candidates in each (platform_a, platform_b) bucket
    # that are in DISCARD get promoted to MABD, ensuring minority-bucket pairs reach Stage 3.
    # Set to 0 to disable (default). Recommended value: 10–20 when using --openai.
    per_bucket_min_mabd: int = 0


@dataclass
class Stage3Config:
    llm_model: str = "gpt-4.1-nano"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    min_confidence: float = 0.70


@dataclass
class MBridgeNetConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    lambda_weight: float = 0.5
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    n_folds: int = 5


def load_config(path: Optional[Path] = None) -> MBridgeNetConfig:
    if path is None:
        path = ROOT / "configs" / "default.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return MBridgeNetConfig(
        stage1=Stage1Config(**raw.get("stage1", {})),
        stage2=Stage2Config(**raw.get("stage2", {})),
        stage3=Stage3Config(**raw.get("stage3", {})),
        lambda_weight=raw.get("fusion", {}).get("lambda_weight", 0.5),
        k_values=raw.get("evaluation", {}).get("k_values", [5, 10, 20, 50]),
        n_folds=raw.get("evaluation", {}).get("n_folds", 5),
    )
