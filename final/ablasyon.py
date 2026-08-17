'''Öznitelik ablasyonu: her mantıklı öznitelik grubunun test başarısına katkısı.

Kümülatif olarak öznitelik ekler (taban -> +pitch -> +jitter/shimmer ->
+kontrast -> +delta2 -> +bant genişliği) ve her seti aynı resmî bölmede
(seed 42) BİRDEN ÇOK kez eğitir. Neden çok koşu? GPU tam deterministik
olmadığından tek koşu ±%1-2 oynayabilir; ortalama ± std gerçek katkıyı verir.

Sonuç: rapora doğrudan girebilecek bir tablo + ablasyon.csv.

Önce öznitelikleri hazırlayın (her set için bir kez, süreç-paralel):
    python final/hizli_cikarma.py --sadece 32 200 --islemler 14
Not: hizli_cikarma varsayılan (tüm bayraklar açık) seti çıkarır; ablasyon
alt-setleri de bu tam setin parçalarını yeniden kullanamaz (fingerprint
farklı), bu yüzden bu betik her set için eksik özniteliği kendi çıkarır
(ilk sette yavaş, sonra önbellekten).

Örnek:
    python final/ablasyon.py --kosu 5 --feature-workers 16
'''

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.dataset import Standardizer  # noqa: E402
from final.features import IntervalConfig, extract_interval_series  # noqa: E402
from final.models import OptimSettings, RNNConfig, SeqRNN  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import compute_metrics  # noqa: E402
from ser.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger(__name__)

# Kümülatif ablasyon setleri: her satır bir öncekine bir öznitelik grubu ekler.
SETLER = [
    ('taban',      dict(use_pitch=False, use_jitter=False, use_contrast=False, use_delta2=False, use_bandwidth=False)),
    ('+pitch',     dict(use_pitch=True,  use_jitter=False, use_contrast=False, use_delta2=False, use_bandwidth=False)),
    ('+jit/shim',  dict(use_pitch=True,  use_jitter=True,  use_contrast=False, use_delta2=False, use_bandwidth=False)),
    ('+kontrast',  dict(use_pitch=True,  use_jitter=True,  use_contrast=True,  use_delta2=False, use_bandwidth=False)),
    ('+delta2',    dict(use_pitch=True,  use_jitter=True,  use_contrast=True,  use_delta2=True,  use_bandwidth=False)),
    ('+bant',      dict(use_pitch=True,  use_jitter=True,  use_contrast=True,  use_delta2=True,  use_bandwidth=True)),
]

# Model: resmî kazananın ayarları — tüm setlerde AYNI, ki fark yalnız özniteliğe bağlı olsun.
MODEL = RNNConfig(rnn_type='gru', hidden_size=192, num_layers=2, bidirectional=True,
                  dropout=0.3, pooling='mean',
                  optim=OptimSettings(batch_size=64, learning_rate=1e-3,
                                      weight_decay=1e-4, patience=8))


def on_cikar_paralel(yollar, cfg_sozluk, cache_root, islemler):
    '''Bir öznitelik setini süreç-paralel çıkarır (hizli_cikarma altyapısını yeniden kullanır).

    pyin gibi pahalı işlemleri thread yerine 14 çekirdeğe böler; eksik olmayan
    dosyalar atlanır, böylece ikinci kez çağrıldığında saniyeler sürer.
    '''

    from concurrent.futures import ProcessPoolExecutor
    from final.dataset import feature_cache_path
    from final.features import IntervalConfig
    from final.hizli_cikarma import _cikar, _isci_kur

    fp = IntervalConfig(**cfg_sozluk).fingerprint
    eksik = [(y, cfg_sozluk) for y in yollar
             if not feature_cache_path(y, cache_root, fp).is_file()]
    if not eksik:
        return
    log.info('  ön-çıkarma (paralel): %d dosya, %d süreç', len(eksik), islemler)
    with ProcessPoolExecutor(max_workers=islemler,
                             initializer=_isci_kur, initargs=(cache_root,)) as havuz:
        for _ in havuz.map(_cikar, eksik, chunksize=64):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--kosu', type=int, default=5,
                        help='Her set kaç kez eğitilsin (ortalama için).')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--cache-root', default='data/cache/final')
    parser.add_argument('--out', default='final/deneyler/ablasyon')
    parser.add_argument('--feature-workers', type=int, default=8)
    parser.add_argument('--islemler', type=int, default=14,
                        help='Ön-çıkarma için paralel süreç sayısı.')
    parser.add_argument('--max-epochs', type=int, default=60)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = pd.read_csv(args.manifest)
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    out_dir = ensure_dir(args.out)
    defter = Path(out_dir) / 'ablasyon.csv'
    satirlar = []
    taban_acc = None

    tum_yollar = pd.concat([tr, va, te])['path'].astype(str).tolist()

    for ad, bayrak in SETLER:
        fcfg = IntervalConfig(n_intervals=32, interval_ms=200, **bayrak)
        # ÖN-ÇIKARMA (süreç-paralel): bu setin öznitelikleri önbellekte yoksa,
        # 14 çekirdekle hızlıca doldur. Böylece aşağıdaki thread'li okuma saf
        # önbellek okuması olur, pyin beklemesi olmaz.
        on_cikar_paralel(tum_yollar, dict(fcfg.__dict__), args.cache_root, args.islemler)
        # Öznitelikler artık hazır -> hızlı okuma.
        T = _feature_folds(folds, fcfg, extract_interval_series, args.cache_root,
                           workers=args.feature_workers, cache={})
        tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
        olcek = Standardizer.fit(tx, feature_axis=2)
        tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

        accs, f1s = [], []
        for k in range(args.kosu):
            torch.manual_seed(k)
            torch.cuda.manual_seed_all(k)
            model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)
            res = train_with_early_stopping(
                model, tx_s, ty, vx_s, vy, MODEL.optim,
                num_classes=NUM_CLASSES, device=device,
                max_epochs=args.max_epochs, seed=k)
            _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)
            m = compute_metrics(ey, prob.argmax(axis=1))
            accs.append(m['accuracy'])
            f1s.append(m['macro_f1'])

        acc_ort, acc_std = float(np.mean(accs)), float(np.std(accs))
        f1_ort = float(np.mean(f1s))
        if taban_acc is None:
            taban_acc = acc_ort
        katki = acc_ort - taban_acc
        satirlar.append({
            'set': ad, 'boyut': fcfg.feature_dim,
            'test_acc_ort': round(acc_ort, 4), 'test_acc_std': round(acc_std, 4),
            'test_f1_ort': round(f1_ort, 4),
            'tabana_katki': round(katki, 4),
        })
        log.info('[%s] boyut=%d | test acc=%.4f±%.4f  macro-F1=%.4f | tabana katkı %+.4f',
                 ad, fcfg.feature_dim, acc_ort, acc_std, f1_ort, katki)

    # Tabloyu yaz + ekrana bas.
    with open(defter, 'w', newline='', encoding='utf-8') as f:
        yazici = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        yazici.writeheader()
        yazici.writerows(satirlar)

    print(f'\n===== ÖZNİTELİK ABLASYONU ({args.kosu} koşu ortalaması, seed 42) =====')
    print(f'{"set":12s} {"boyut":>5s} {"test acc":>12s} {"macro-F1":>9s} {"katkı":>8s}')
    for s in satirlar:
        print(f'{s["set"]:12s} {s["boyut"]:5d} '
              f'{s["test_acc_ort"]:.4f}±{s["test_acc_std"]:.4f} '
              f'{s["test_f1_ort"]:8.4f} {s["tabana_katki"]:+8.4f}')
    print(f'\nDefter: {defter}')


if __name__ == '__main__':
    main()
