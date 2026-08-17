'''Training-only batch augmentations for the improvement stage.

Both transforms operate on already-standardized feature batches, are applied
only to training batches (never validation/test), and require no pretrained
model: SpecAugment-style masking (Park et al., 2019) for the mel-CNN and
additive Gaussian noise for the interval series.
'''

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecMask:
    '''Zero out random frequency and time stripes of mel images [B, mels, T].

    Masked regions are set to 0, the post-standardization mean.
    '''

    freq_masks: int = 2
    freq_width: int = 8
    time_masks: int = 2
    time_width: int = 16

    def validate(self) -> None:
        if min(self.freq_masks, self.freq_width, self.time_masks, self.time_width) < 0:
            raise ValueError(f'Mask settings must be non-negative: {self}.')

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f'Expected [B, mels, T] batch, got {tuple(batch.shape)}.')
        batch = batch.clone()
        n, n_mels, n_frames = batch.shape
        for _ in range(self.freq_masks):
            widths = torch.randint(0, self.freq_width + 1, (n,), device=batch.device)
            starts = (torch.rand(n, device=batch.device)
                      * (n_mels - widths).clamp(min=0)).long()
            for i in range(n):
                batch[i, starts[i] : starts[i] + widths[i], :] = 0.0
        for _ in range(self.time_masks):
            widths = torch.randint(0, self.time_width + 1, (n,), device=batch.device)
            starts = (torch.rand(n, device=batch.device)
                      * (n_frames - widths).clamp(min=0)).long()
            for i in range(n):
                batch[i, :, starts[i] : starts[i] + widths[i]] = 0.0
        return batch


@dataclass(frozen=True)
class FeatureNoise:
    '''Additive Gaussian noise on standardized interval series [B, T, D].'''

    std: float = 0.1

    def validate(self) -> None:
        if self.std < 0.0:
            raise ValueError(f'Noise std must be non-negative: {self.std}.')

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f'Expected [B, T, D] batch, got {tuple(batch.shape)}.')
        return batch + torch.randn_like(batch) * self.std
