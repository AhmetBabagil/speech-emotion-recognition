"""Training loop for the deep models (CNN on log-mel, wav2vec2 on waveform).

Handles class-weighted loss, label smoothing, cosine LR, AMP (GPU), gradient
clipping, early stopping on the validation monitor metric, best-checkpoint
saving, and a final test-set report. Returns the test metrics dict.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import NUM_CLASSES
from .config import Config
from .data import SERDataset, class_weights, prepare_splits
from .evaluate import evaluate_torch, report, compute_metrics
from .models import build_model
from .utils import get_logger, set_seed, get_device, ensure_dir, count_params

log = get_logger(__name__)


def _make_loaders(cfg, train_df, val_df, test_df, device):
    import torch
    from torch.utils.data import DataLoader

    mode = "waveform" if cfg.model.name.lower() == "wav2vec2" else "logmel"
    pin = device.type == "cuda"
    common = dict(num_workers=cfg.train.num_workers, pin_memory=pin)
    # Seeded generator -> reproducible shuffle order even with num_workers > 0.
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        SERDataset(train_df, cfg, mode=mode, train=True),
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=False,
        generator=g, **common,
    )
    val_loader = DataLoader(
        SERDataset(val_df, cfg, mode=mode, train=False),
        batch_size=cfg.train.batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        SERDataset(test_df, cfg, mode=mode, train=False),
        batch_size=cfg.train.batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, test_loader


def _amp_tools(cfg, device):
    import torch

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def autocast():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    return scaler, autocast


def _train_one_epoch(model, loader, criterion, optimizer, scaler, autocast, device, grad_clip):
    import torch

    model.train()
    total, running = 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = yb.size(0)
        running += loss.item() * bs
        total += bs
    return running / max(total, 1)


def train_torch(cfg: Config, device=None) -> dict:
    import torch
    import torch.nn as nn

    set_seed(cfg.train.seed)
    device = device or get_device()
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    cfg.save(out_dir / "config.yaml")

    df = pd.read_csv(cfg.data.manifest)
    train_df, val_df, test_df = prepare_splits(df, cfg.data, cfg.train.seed)
    for name, part in (("Training", train_df), ("Validation", val_df), ("Test", test_df)):
        if len(part) == 0:
            raise ValueError(f"{name} set is empty -- check split fractions / corpora.")

    train_loader, val_loader, test_loader = _make_loaders(cfg, train_df, val_df, test_df, device)

    model = build_model(cfg, NUM_CLASSES).to(device)
    log.info("Model '%s' with %s trainable params", cfg.model.name, f"{count_params(model):,}")

    weights = class_weights(train_df, cfg.train.class_weighting).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler, autocast = _amp_tools(cfg, device)

    best_score, best_epoch, best_path = -np.inf, -1, out_dir / "best.pt"
    history = []
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        train_loader.dataset.set_epoch(epoch)  # vary augmentation, reproducibly
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, autocast, device, cfg.train.grad_clip
        )
        y_true, y_pred, _ = evaluate_torch(model, val_loader, device)
        vmetrics = compute_metrics(y_true, y_pred)
        score = vmetrics[cfg.train.monitor]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_accuracy": vmetrics["accuracy"], "val_macro_f1": vmetrics["macro_f1"]})
        log.info("epoch %3d | loss %.4f | val_acc %.4f | val_macroF1 %.4f%s",
                 epoch, train_loss, vmetrics["accuracy"], vmetrics["macro_f1"],
                 "  *" if score > best_score else "")

        if score > best_score:
            best_score, best_epoch, epochs_no_improve = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": cfg.to_dict()}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.early_stop_patience:
                log.info("Early stopping at epoch %d (best epoch %d, %s=%.4f)",
                         epoch, best_epoch, cfg.train.monitor, best_score)
                break

    # Restore best and evaluate on test.
    if best_path.exists():
        # weights_only=False: our checkpoint also stores the config dict.
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        log.info("Loaded best checkpoint from epoch %d", state["epoch"])

    y_true, y_pred, _ = evaluate_torch(model, test_loader, device)
    metrics = report(y_true, y_pred, out_dir, prefix="test",
                     title=f"{cfg.experiment} (test)")
    metrics["best_epoch"] = best_epoch
    metrics["val_best_" + cfg.train.monitor] = best_score
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    return metrics


def train_baseline(cfg: Config, kind: str = "svm") -> dict:
    """Train + evaluate the classical MFCC-statistics baseline (sklearn)."""
    from .data import mfcc_feature_matrix
    from .models import build_baseline

    set_seed(cfg.train.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    cfg.save(out_dir / "config.yaml")

    df = pd.read_csv(cfg.data.manifest)
    train_df, val_df, test_df = prepare_splits(df, cfg.data, cfg.train.seed)
    for name, part in (("Training", train_df), ("Test", test_df)):
        if len(part) == 0:
            raise ValueError(f"{name} set is empty -- check split fractions / corpora.")
    # Fit on the TRAIN split only (no val), so the baseline and the deep models
    # learn from the same data and the comparison is fair + leak-free. The
    # StandardScaler inside the pipeline is therefore fit on train statistics only.
    fit_df = train_df

    log.info("Extracting MFCC-statistics features ...")
    X_train, y_train = mfcc_feature_matrix(fit_df, cfg)
    X_test, y_test = mfcc_feature_matrix(test_df, cfg)

    pipe = build_baseline(kind)
    log.info("Fitting baseline '%s' on %d samples (dim=%d) ...", kind, len(y_train), X_train.shape[1])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = report(y_test, y_pred, out_dir, prefix="test",
                     title=f"{cfg.experiment} baseline-{kind} (test)")
    try:
        import joblib
        joblib.dump(pipe, out_dir / "baseline.joblib")
    except Exception as e:
        log.warning("Could not save baseline model: %s", e)
    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    return metrics
