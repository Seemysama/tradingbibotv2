from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch


def _detect_device() -> str:
    """Pick the best available device (prefers Apple MPS, then CUDA, then CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Settings:
    DATA_PATH: Path = Path("data/historical")
    TIMEFRAME: str = "1m"
    PAIRS: List[str] = None
    DEVICE: str = _detect_device()

    def __post_init__(self) -> None:
        if self.PAIRS is None:
            self.PAIRS = ["BTC/USDT"]


settings = Settings()
