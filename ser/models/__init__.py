"""Models: classical MFCC baseline, log-mel CNN, wav2vec2 transfer learning."""

from .cnn import MelCNN
from .baseline import build_baseline


def build_model(cfg, num_classes: int):
    """Factory for the torch models (the sklearn baseline is built separately)."""
    name = cfg.model.name.lower()
    if name == "cnn":
        return MelCNN(
            num_classes=num_classes,
            in_ch=1,
            channels=tuple(cfg.model.cnn_channels),
            dropout=cfg.model.dropout,
        )
    if name == "wav2vec2":
        from .wav2vec2 import Wav2Vec2Classifier

        return Wav2Vec2Classifier(
            num_classes=num_classes,
            pretrained_name=cfg.model.pretrained_name,
            freeze_feature_encoder=cfg.model.freeze_feature_encoder,
            dropout=cfg.model.dropout,
        )
    raise ValueError(f"Unknown model.name={cfg.model.name!r} (expected 'cnn' or 'wav2vec2')")


__all__ = ["MelCNN", "build_baseline", "build_model"]
