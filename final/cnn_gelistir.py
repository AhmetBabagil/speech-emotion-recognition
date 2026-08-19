# Yöntem 1 (CNN) geliştirme deneyleri: hangi teknik doğruluğu artırıyor?
#
# Her varyant, aynı mel öznitelikleri üzerinde 5 kez eğitilir (GPU tam deterministik olmadığı için tek koşu güvenilmez). SEÇİM ÖLÇÜTÜ geçerleme (validation) macro-F1'idir — test'e bakarak seçmek sızıntı olur; test yalnızca bilgi amaçlı raporlanır. Kazanan (geçerlemede en iyi) varyant, resmî modele terfi adayıdır.
#
# Denenen teknikler:
# * taban          : mevcut kazanan CNN, artırma yok (dürüst 5-koşu tabanı)
# * specaug        : SpecAugment maskeleme (mevcut geliştirme)
# * specaug_guclu  : daha çok/geniş maske
# * label_smooth   : etiket yumuşatma 0.1 (aşırı güvene ceza)
# * mixup          : mixup alpha 0.2 (örnek+etiket karıştırma)
# * specaug_ls     : specaug + label smoothing (birlikte)
# * derin          : 4 bloklu daha derin ağ (32-64-128-256)
# * genis_drop     : daha geniş + yüksek dropout (48-96-192, drop 0.4)
#
# Örnek:
# python final/cnn_gelistir.py --kosu 5
# python final/cnn_gelistir.py --kosu 1 --sadece taban specaug   # hızlı deneme

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.augment import SpecMask  # noqa: E402
from final.dataset import Standardizer  # noqa: E402
from final.features import MelImageConfig, extract_mel_image  # noqa: E402
from final.models import CNNConfig, MelCNN, OptimSettings  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import compute_metrics  # noqa: E402

# Kazanan CNN yapılandırması (winner.json ile aynı).
OPT = OptimSettings(batch_size=16, learning_rate=1e-3, weight_decay=1e-4, patience=8)
TABAN_CFG = CNNConfig(channels=(32, 64, 128), dropout=0.3, optim=OPT)


def varyantlar():  # (ad, model_cfg, transform, label_smoothing, mixup_alpha) listesi.
    return [
        ('taban',         TABAN_CFG, None, 0.0, 0.0),
        ('specaug',       TABAN_CFG, SpecMask(), 0.0, 0.0),
        ('specaug_guclu', TABAN_CFG,
         SpecMask(freq_masks=2, freq_width=12, time_masks=2, time_width=24), 0.0, 0.0),
        ('label_smooth',  TABAN_CFG, None, 0.1, 0.0),
        ('mixup',         TABAN_CFG, None, 0.0, 0.2),
        ('specaug_ls',    TABAN_CFG, SpecMask(), 0.1, 0.0),
        ('derin',         replace(TABAN_CFG, channels=(32, 64, 128, 256)), SpecMask(), 0.0, 0.0),
        ('genis_drop',    replace(TABAN_CFG, channels=(48, 96, 192), dropout=0.4),
         SpecMask(), 0.0, 0.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kosu', type=int, default=5)
    parser.add_argument('--sadece', nargs='+', default=None,
                        help='yalnız bu varyant adlarını koştur')
    parser.add_argument('--feature-workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    # Mel öznitelikleri (önbellekten anında gelir; feature_axis=1).
    fcfg = MelImageConfig()
    T = _feature_folds(folds, fcfg, extract_mel_image, 'data/cache/final',
                       workers=args.feature_workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=1)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)
    print(f'Mel şekli: train={tx_s.shape}  (n_mels×kare)')

    secili = varyantlar()
    if args.sadece:
        secili = [v for v in secili if v[0] in set(args.sadece)]

    satirlar = []
    for ad, cfg, tf, ls, mix in secili:
        vfs, accs, f1s = [], [], []
        t0 = time.perf_counter()
        for k in range(args.kosu):
            outcome = train_with_early_stopping(
                MelCNN(NUM_CLASSES, cfg), tx_s, ty, vx_s, vy, cfg.optim,
                num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k,
                amp=True, train_transform=tf, label_smoothing=ls, mixup_alpha=mix)
            vfs.append(outcome.validation_metrics['macro_f1'])
            _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)
            m = compute_metrics(ey, prob.argmax(axis=1))
            accs.append(m['accuracy']); f1s.append(m['macro_f1'])
        sn = (time.perf_counter() - t0) / max(args.kosu, 1)
        satirlar.append((ad, float(np.mean(vfs)), float(np.mean(accs)),
                         float(np.std(accs)), float(np.mean(f1s)), sn))
        print(f'  [{ad:14s}] val_macroF1={np.mean(vfs):.4f}  '
              f'test_acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
              f'test_F1={np.mean(f1s):.4f}  ({sn:.0f} sn/koşu)')

    # Sonuçları JSON'a yaz (arka plan koşusunu sonradan okumak için).
    import json
    kayit = [{'varyant': ad, 'val_macro_f1': vf, 'test_acc': acc,
              'test_acc_std': std, 'test_macro_f1': f1, 'sn_kosu': sn}
             for ad, vf, acc, std, f1, sn in satirlar]
    Path('final/outputs/cremad/cnn').mkdir(parents=True, exist_ok=True)
    Path('final/outputs/cremad/cnn/gelistirme_deney.json').write_text(
        json.dumps({'kosu': args.kosu, 'sonuclar': kayit}, indent=2), encoding='utf-8')

    # Geçerleme macro-F1'e göre sırala (dürüst seçim ölçütü).
    satirlar.sort(key=lambda r: r[1], reverse=True)
    print('\n===== CNN GELİŞTİRME (5 koşu ort., geçerlemeye göre sıralı) =====')
    print(f'{"varyant":15s} {"val macroF1":>12s} {"test acc":>14s} {"test F1":>9s}')
    for ad, vf, acc, std, f1, _sn in satirlar:
        yildiz = '  <-- en iyi (val)' if (ad, vf) == (satirlar[0][0], satirlar[0][1]) else ''
        print(f'{ad:15s} {vf:12.4f} {acc:.4f}±{std:.4f} {f1:8.4f}{yildiz}')
    print('\nSeçim geçerlemeye göre; test yalnız bilgi. Taban ile kıyasla.')


if __name__ == '__main__':
    main()
