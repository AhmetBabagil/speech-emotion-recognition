'''Predict the emotion of one or more WAV files with a trained final model.

Loads a checkpoint saved by run_experiment.py / improve.py (feature settings,
standardizer and weights travel together), so predictions match the training
pipeline exactly.

Examples:
    python final/predict.py data/raw/cremad/AudioWAV/1001_DFA_ANG_XX.wav
    python final/predict.py clip.wav --model final/outputs/cremad/rnn/winner_model.pt
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from final.dataset import Standardizer  # noqa: E402
from final.features import (  # noqa: E402
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import CNNConfig, MelCNN, OptimSettings, RNNConfig, SeqRNN  # noqa: E402
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # noqa: E402


def load_checkpoint(path: str | Path, device: torch.device):
    '''Rebuild (model, feature_config, extract_fn, standardizer) from disk.'''

    # Trusted local file written by our own training scripts; it contains
    # numpy arrays, so the weights-only loader cannot be used.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    feature_values = checkpoint['feature_config']
    model_values = dict(checkpoint['model_config'])
    optim = OptimSettings(**model_values.pop('optim'))

    if 'n_mels' in feature_values:
        feature_cfg = MelImageConfig(**feature_values)
        extract_fn = extract_mel_image
        model = MelCNN(
            NUM_CLASSES,
            CNNConfig(channels=tuple(model_values['channels']),
                      dropout=model_values['dropout'], optim=optim),
        )
    else:
        feature_cfg = IntervalConfig(**feature_values)
        extract_fn = extract_interval_series
        model = SeqRNN(
            feature_cfg.feature_dim,
            NUM_CLASSES,
            RNNConfig(**model_values, optim=optim),
        )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    standardizer = Standardizer(
        mean=np.asarray(checkpoint['standardizer_mean']),
        scale=np.asarray(checkpoint['standardizer_scale']),
        feature_axis=int(checkpoint['feature_axis']),
    )
    return model, feature_cfg, extract_fn, standardizer


def predict(paths, model, feature_cfg, extract_fn, standardizer, device):
    features = np.stack([extract_fn(path, feature_cfg) for path in paths])
    features = standardizer.transform(features)
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('audio', nargs='+', help='Bir veya daha fazla WAV dosyası.')
    parser.add_argument(
        '--model', default='final/outputs/cremad/cnn/winner_model.pt'
    )
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model, feature_cfg, extract_fn, standardizer = load_checkpoint(args.model, device)
    probabilities = predict(
        args.audio, model, feature_cfg, extract_fn, standardizer, device
    )
    for path, row in zip(args.audio, probabilities):
        order = np.argsort(row)[::-1]
        ranking = ', '.join(
            f'{CANONICAL_EMOTIONS[i]}={row[i]:.3f}' for i in order[:3]
        )
        print(f'{path}: {CANONICAL_EMOTIONS[order[0]]}  ({ranking})')


if __name__ == '__main__':
    main()
