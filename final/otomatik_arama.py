# Otomatik rastgele hiperparametre araması — MAKSİMUM GÜÇ sürümü.
#
# Rastgele arama, ızgaradan daha verimlidir (Bergstra & Bengio, 2012): her denemede uzayın yeni bir noktası yoklanır. Bu sürüm donanımı sonuna kadar kullanır:
#
# 1. ÖN-ÇIKARMA: denenecek tüm aralık düzenlerinin öznitelikleri baştan, çok
# iş parçacığıyla çıkarılır (bir kez; sonrası önbellek). Böylece deneme
# döngüsü sırasında GPU asla CPU'yu beklemez.
# 2. PARALEL DENEME: --paralel N ile aynı anda N model birden eğitilir.
# Modeller küçük olduğu için RTX 5080 üç-dört eğitimi rahat taşır.
# 3. Her sonuç ANINDA arama_defteri.csv'ye yazılır (çökse bile kayıp olmaz).
#
# Test disiplini: bu betik TEST'E DOKUNMAZ — hakem yalnızca geçerleme kümesi.
#
# Örnek (tam güç):
# python final/otomatik_arama.py --denemeler 100 --paralel 3 --feature-workers 24

from __future__ import annotations

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Örnekleme uzayı: resmî aramanın denemediği uçlar da bilinçli olarak dahil.
ARALIK_SAYILARI = (16, 24, 32, 40, 48)
ARALIK_GENISLIKLERI = (150, 200, 250, 300, 400)      # ms
GIZLI_BOYUTLAR = (96, 128, 192, 256, 320)
KATMANLAR = (1, 2, 3)
DROPOUTLAR = (0.1, 0.2, 0.3, 0.4, 0.5)
HAVUZLAMALAR = ('last', 'mean', 'max', 'attn')
BATCHLER = (32, 64, 128)
WEIGHT_DECAYLER = (0.0, 1e-4, 1e-3)

# Her işçi süreçte bir kez kurulan küresel durum (initializer doldurur).
_FOLDS = None
_CACHE_ROOT = None
_FEATURE_CACHE: dict = {}
_MAX_EPOCHS = 60


def rastgele_aday(rng: random.Random):  # Uzaydan tek bir rastgele (aralık düzeni, model ayarı) ikilisi çeker.

    from final.features import IntervalConfig
    from final.models import OptimSettings, RNNConfig

    feature_cfg = IntervalConfig(
        n_intervals=rng.choice(ARALIK_SAYILARI),
        interval_ms=rng.choice(ARALIK_GENISLIKLERI),
    )
    lr = 10 ** rng.uniform(-3.52, -2.52)   # 3e-4 .. 3e-3, log-düzgün
    model_cfg = RNNConfig(
        rnn_type=rng.choice(('gru', 'lstm')),
        hidden_size=rng.choice(GIZLI_BOYUTLAR),
        num_layers=rng.choice(KATMANLAR),
        bidirectional=rng.choice((True, False)),
        dropout=rng.choice(DROPOUTLAR),
        pooling=rng.choice(HAVUZLAMALAR),
        optim=OptimSettings(
            batch_size=rng.choice(BATCHLER),
            learning_rate=round(lr, 5),
            weight_decay=rng.choice(WEIGHT_DECAYLER),
            patience=8,
        ),
    )
    return feature_cfg, model_cfg


def _isci_kur(manifest_path: str, seed: int, cache_root: str, max_epochs: int) -> None:  # Her paralel işçi süreç başlarken BİR KEZ çalışır: bölmeyi kurar.

    global _FOLDS, _CACHE_ROOT, _MAX_EPOCHS
    import pandas as pd
    from final.pipeline import SplitSettings
    from ser.data.splits import prepare_splits

    manifest = pd.read_csv(manifest_path)
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    train_df, val_df, _test = prepare_splits(manifest, ayar, seed=seed)
    _FOLDS = {'train': train_df, 'val': val_df}
    _CACHE_ROOT = cache_root
    _MAX_EPOCHS = max_epochs


def _deneme_kos(girdi):  # Tek bir adayı eğitir; sonuç satırını döndürür (işçi süreçte koşar).

    k, feature_cfg, model_cfg = girdi
    import torch
    from final.dataset import Standardizer
    from final.features import extract_interval_series
    from final.models import SeqRNN, count_parameters
    from final.pipeline import _feature_folds
    from final.training import train_with_early_stopping
    from ser.constants import NUM_CLASSES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Öznitelikler ön-çıkarma sayesinde diskte hazır; buradaki yükleme saf okuma.
    tensorlar = _feature_folds(_FOLDS, feature_cfg, extract_interval_series,
                               _CACHE_ROOT, workers=8, cache=_FEATURE_CACHE)
    train_x, train_y = tensorlar['train']
    val_x, val_y = tensorlar['val']
    olcek = Standardizer.fit(train_x, feature_axis=2)
    model = SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)
    basla = time.perf_counter()
    sonuc = train_with_early_stopping(
        model, olcek.transform(train_x), train_y,
        olcek.transform(val_x), val_y, model_cfg.optim,
        num_classes=NUM_CLASSES, device=device,
        max_epochs=_MAX_EPOCHS, seed=42,
    )
    saniye = time.perf_counter() - basla
    m = sonuc.validation_metrics
    return [k, feature_cfg.n_intervals, feature_cfg.interval_ms,
            model_cfg.rnn_type, model_cfg.hidden_size, model_cfg.num_layers,
            model_cfg.bidirectional, model_cfg.dropout, model_cfg.pooling,
            model_cfg.optim.batch_size, model_cfg.optim.learning_rate,
            model_cfg.optim.weight_decay, count_parameters(model),
            sonuc.best_epoch, round(m['accuracy'], 4),
            round(m['macro_f1'], 4), round(saniye, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--denemeler', type=int, default=60)
    parser.add_argument('--paralel', type=int, default=3,
                        help='Aynı anda kaç deneme eğitilsin (GPU paylaşımlı).')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--cache-root', default='data/cache/final')
    parser.add_argument('--out', default='final/deneyler/rastgele_arama')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ornekleme-seed', type=int, default=1)
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--feature-workers', type=int, default=24,
                        help='Ön-çıkarma iş parçacığı sayısı.')
    args = parser.parse_args()

    import pandas as pd
    from final.features import extract_interval_series
    from final.pipeline import SplitSettings, _feature_folds
    from ser.data.splits import prepare_splits
    from ser.utils import ensure_dir, get_logger

    log = get_logger(__name__)

    # 1) Adayları baştan örnekle (hangi aralık düzenleri lazım, bilelim).
    rng = random.Random(args.ornekleme_seed)
    adaylar = [rastgele_aday(rng) for _ in range(args.denemeler)]

    # 2) ÖN-ÇIKARMA: gereken her aralık düzeninin öznitelikleri tek seferde.
    manifest = pd.read_csv(args.manifest)
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    train_df, val_df, _test = prepare_splits(manifest, ayar, seed=args.seed)
    folds = {'train': train_df, 'val': val_df}
    gereken = {f for f, _ in adaylar}
    log.info('Ön-çıkarma: %d farklı aralık düzeni, %d iş parçacığı',
             len(gereken), args.feature_workers)
    ge_basla = time.perf_counter()
    for i, feature_cfg in enumerate(sorted(gereken,
                                           key=lambda c: (c.n_intervals, c.interval_ms)), 1):
        _feature_folds(folds, feature_cfg, extract_interval_series,
                       args.cache_root, workers=args.feature_workers, cache={})
        log.info('  [%d/%d] %dx%dms hazır', i, len(gereken),
                 feature_cfg.n_intervals, feature_cfg.interval_ms)
    log.info('Ön-çıkarma bitti: %.0f sn', time.perf_counter() - ge_basla)

    # 3) PARALEL DENEME DÖNGÜSÜ.
    out_dir = ensure_dir(args.out)
    defter = Path(out_dir) / 'arama_defteri.csv'
    yeni_dosya = not defter.exists()
    en_iyi_f1 = -1.0
    bitti = 0
    tur_basla = time.perf_counter()
    with open(defter, 'a', newline='', encoding='utf-8') as f:
        yazici = csv.writer(f)
        if yeni_dosya:
            yazici.writerow(['deneme', 'n_intervals', 'interval_ms', 'rnn_type',
                             'hidden', 'layers', 'bidir', 'dropout', 'pooling',
                             'batch', 'lr', 'weight_decay', 'parametre',
                             'best_epoch', 'val_accuracy', 'val_macro_f1', 'saniye'])
        gorevler = [(k, fc, mc) for k, (fc, mc) in enumerate(adaylar, start=1)]
        with ProcessPoolExecutor(
            max_workers=args.paralel,
            initializer=_isci_kur,
            initargs=(args.manifest, args.seed, args.cache_root, args.max_epochs),
        ) as havuz:
            for satir in (fut.result() for fut in
                          as_completed(havuz.submit(_deneme_kos, g) for g in gorevler)):
                yazici.writerow(satir)
                f.flush()
                bitti += 1
                f1 = satir[15]
                isaret = ''
                if f1 > en_iyi_f1:
                    en_iyi_f1 = f1
                    isaret = '  <-- YENİ LİDER'
                log.info('[%d/%d] deneme#%d %dx%dms %s h%d L%d %s | val F1=%.4f%s',
                         bitti, len(gorevler), satir[0], satir[1], satir[2],
                         satir[3], satir[4], satir[5], satir[8], f1, isaret)
    log.info('Deneme döngüsü: %.0f sn (%d deneme, paralel=%d)',
             time.perf_counter() - tur_basla, len(gorevler), args.paralel)

    # 4) Özet.
    tablo = pd.read_csv(defter).sort_values('val_macro_f1', ascending=False)
    print('\n===== EN İYİ 5 ADAY (geçerleme macro-F1) =====')
    print(tablo.head(5).to_string(index=False))
    resmi = 0.5638
    lider = float(tablo.iloc[0]['val_macro_f1'])
    if lider > resmi:
        print(f'\nLider ({lider:.4f}) resmî kazananı ({resmi:.4f}) GEÇTİ — '
              'test teyidi için run_experiment ile bilinçli koşu yapın.')
    else:
        print(f'\nLider ({lider:.4f}) resmî kazananı ({resmi:.4f}) geçemedi — '
              'mevcut ayarlar savunuldu.')


if __name__ == '__main__':
    main()
