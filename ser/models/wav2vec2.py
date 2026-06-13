"""Transfer learning: a classification head on top of a pretrained wav2vec2.

Input : [B, num_samples] raw waveform (16 kHz, already normalized by the dataset)
Output: [B, num_classes] logits

The pretrained convolutional feature encoder can be frozen (recommended for small
SER datasets) while the transformer layers + head are fine-tuned. Requires
``transformers`` (``pip install -e .[transfer]``). This path is GPU-oriented.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_name: str = "facebook/wav2vec2-base",
                 freeze_feature_encoder: bool = True, dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "wav2vec2 model requires `transformers`. Install with "
                "`pip install -e .[transfer]` or `pip install transformers`."
            ) from e

        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_name)
        if freeze_feature_encoder:
            self.backbone.feature_extractor._freeze_parameters()
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_values).last_hidden_state  # [B, T', H]
        pooled = out.mean(dim=1)  # mean pool over time
        return self.head(pooled)
