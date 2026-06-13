"""Shared utilities: device selection, seeding, logging, small helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

_LOG_CONFIGURED = False


def get_logger(name: str = "ser") -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(prefer: str = "auto") -> "object":
    """Return a torch.device, auto-detecting CUDA. ``prefer`` may be
    'auto' | 'cuda' | 'cpu'. Falls back to CPU with a warning if CUDA is
    requested but unavailable."""
    import torch

    log = get_logger()
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info("Using CUDA device: %s", name)
        return torch.device("cuda")
    if prefer == "cuda":
        log.warning("CUDA requested but not available -- falling back to CPU.")
    else:
        log.info("CUDA not available -- using CPU.")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
