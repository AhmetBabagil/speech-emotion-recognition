'''Demo: kaydedilmiş final modellerini WAV dosyalarında çalıştırır,
iki yöntemin çıktısını yan yana gösterir.

Yönergenin 8. bölümünün birebir karşılığı: girdiler ya iyi/kötü örnek
dizininden RASTGELE seçilir ya da komut satırında ADIYLA/YOLUYLA verilir;
her girdi iki yöntemden geçirilip sonuçlarıyla gösterilir.

Örnekler:
    python final/demo.py --rastgele 3
    python final/demo.py final/demo_ornekleri/1091_IEO_FEA_HI.wav
    python final/demo.py --dizin final/demo_ornekleri --rastgele 5
'''

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from final.predict import load_checkpoint, predict  # noqa: E402
from ser.constants import CANONICAL_EMOTIONS, CREMAD_CODE_TO_CANONICAL  # noqa: E402

# Test kümesinden seçilmiş 6 iyi + 4 kötü örneğin durduğu varsayılan dizin.
DEFAULT_DIR = Path('final/demo_ornekleri')

# Her yöntem için tercih sırası: geliştirme aşamasının modeli varsa onu,
# yoksa arama kazananını kullan.
MODEL_PATHS = {
    'Yöntem 1 (CNN)': (
        'final/outputs/cremad/cnn/improved_model.pt',
        'final/outputs/cremad/cnn/winner_model.pt',
    ),
    'Yöntem 2 (BiGRU)': (
        'final/outputs/cremad/rnn/improved_model.pt',
        'final/outputs/cremad/rnn/winner_model.pt',
    ),
}


def true_label_from_name(path: Path) -> str | None:
    '''CREMA-D dosya adının 3. parçası duygu kodudur (ör. ..._ANG_... -> angry).

    Bu sayede demo, tahminin doğru mu yanlış mı olduğunu (✓/✗) etiketli
    dosyalarda otomatik gösterebilir.
    '''

    parts = path.stem.split('_')
    if len(parts) >= 3:
        return CREMAD_CODE_TO_CANONICAL.get(parts[2].upper())
    return None


def load_methods(device: torch.device) -> dict:
    '''İki yöntemin modellerini diskten yükler.'''

    methods = {}
    for name, candidates in MODEL_PATHS.items():
        # Tercih listesindeki ilk mevcut dosyayı seç.
        chosen = next((p for p in candidates if Path(p).is_file()), None)
        if chosen is None:
            raise FileNotFoundError(
                f'{name} için model bulunamadı; önce final/run_experiment.py çalıştırın.'
            )
        methods[name] = load_checkpoint(chosen, device)
    return methods


def run_demo(paths: list[Path], methods: dict, device: torch.device) -> None:
    '''Her dosyayı iki yöntemden geçirir; tahmin + ✓/✗ + ilk 3 olasılığı basar.'''

    for path in paths:
        truth = true_label_from_name(path)
        print(f'\n{path.name}   (gerçek: {truth or "bilinmiyor"})')
        for name, (model, feature_cfg, extract_fn, standardizer) in methods.items():
            row = predict([path], model, feature_cfg, extract_fn, standardizer, device)[0]
            order = np.argsort(row)[::-1]
            guess = CANONICAL_EMOTIONS[order[0]]
            isaret = '' if truth is None else ('  ✓' if guess == truth else '  ✗')
            top3 = ', '.join(f'{CANONICAL_EMOTIONS[i]}={row[i]:.2f}' for i in order[:3])
            print(f'  {name}: {guess}{isaret}   ({top3})')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('girdi', nargs='*', help='WAV dosya yolları (adıyla çağırma).')
    parser.add_argument('--dizin', default=str(DEFAULT_DIR),
                        help='İyi/kötü örneklerin bulunduğu dizin.')
    parser.add_argument('--rastgele', type=int, default=0,
                        help='Dizinden rastgele bu kadar örnek seç.')
    parser.add_argument('--seed', type=int, help='Rastgele seçim için tohum.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()

    # Girdi listesi: elle verilenler + (istenirse) dizinden rastgele seçilenler.
    paths = [Path(p) for p in args.girdi]
    if args.rastgele > 0:
        havuz = sorted(Path(args.dizin).glob('*.wav'))
        if not havuz:
            parser.error(f'{args.dizin} içinde .wav bulunamadı.')
        rng = random.Random(args.seed)   # seed verilirse seçim tekrarlanabilir
        paths.extend(rng.sample(havuz, min(args.rastgele, len(havuz))))
    if not paths:
        parser.error('En az bir dosya verin ya da --rastgele N kullanın.')
    eksik = [p for p in paths if not p.is_file()]
    if eksik:
        parser.error(f'Dosya bulunamadı: {eksik}')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    methods = load_methods(device)
    run_demo(paths, methods, device)


if __name__ == '__main__':
    main()
