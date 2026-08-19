# Otomatik rastgele hiperparametre araması — MAKSİMUM GÜÇ sürümü.
#
# Rastgele arama, ızgaradan daha verimlidir (Bergstra & Bengio, 2012): her denemede uzayın yeni bir noktası yoklanır. Bu sürüm donanımı sonuna kadar kullanır:
#
# 1. ÖN-ÇIKARMA: denenecek tüm aralık düzenlerinin öznitelikleri baştan, çok iş parçacığıyla çıkarılır (bir kez; sonrası önbellek). Böylece deneme döngüsü sırasında GPU asla CPU'yu beklemez.
# 2. PARALEL DENEME: --paralel N ile aynı anda N model birden eğitilir. Modeller küçük olduğu için RTX 5080 üç-dört eğitimi rahat taşır.
# 3. Her sonuç ANINDA arama_defteri.csv'ye yazılır (çökse bile kayıp olmaz).
#
# Test disiplini: bu betik TEST'E DOKUNMAZ — hakem yalnızca geçerleme kümesi.
#
# Örnek (tam güç): python final/otomatik_arama.py --denemeler 100 --paralel 3 --feature-workers 24

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
import csv  # arama defterini CSV'ye yazmak için
import random  # rastgele örnekleme
import time  # süre ölçümü
from concurrent.futures import ProcessPoolExecutor, as_completed  # paralel deneme + tamamlananları toplama
from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

# Örnekleme uzayı: resmî aramanın denemediği uçlar da bilinçli olarak dahil.
ARALIK_SAYILARI = (16, 24, 32, 40, 48)  # aralık sayısı seçenekleri
ARALIK_GENISLIKLERI = (150, 200, 250, 300, 400)      # ms  # aralık genişliği seçenekleri
GIZLI_BOYUTLAR = (96, 128, 192, 256, 320)  # RNN gizli boyut seçenekleri
KATMANLAR = (1, 2, 3)  # katman sayısı seçenekleri
DROPOUTLAR = (0.1, 0.2, 0.3, 0.4, 0.5)  # dropout seçenekleri
HAVUZLAMALAR = ('last', 'mean', 'max', 'attn')  # havuzlama seçenekleri
BATCHLER = (32, 64, 128)  # yığın boyu seçenekleri
WEIGHT_DECAYLER = (0.0, 1e-4, 1e-3)  # weight decay seçenekleri

# Her işçi süreçte bir kez kurulan küresel durum (initializer doldurur).
_FOLDS = None  # süreç-genel katmanlar
_CACHE_ROOT = None  # süreç-genel önbellek kökü
_FEATURE_CACHE: dict = {}  # süreç-içi öznitelik belleği
_MAX_EPOCHS = 60  # süreç-genel epoch üst sınırı


def rastgele_aday(rng: random.Random):  # Uzaydan tek bir rastgele (aralık düzeni, model ayarı) ikilisi çeker.

    from final.features import IntervalConfig  # (fonksiyon içi import)
    from final.models import OptimSettings, RNNConfig

    feature_cfg = IntervalConfig(  # rastgele aralık düzeni
        n_intervals=rng.choice(ARALIK_SAYILARI),  # rastgele aralık sayısı
        interval_ms=rng.choice(ARALIK_GENISLIKLERI),  # rastgele aralık genişliği
    )
    lr = 10 ** rng.uniform(-3.52, -2.52)   # 3e-4 .. 3e-3, log-düzgün
    model_cfg = RNNConfig(  # rastgele model ayarı
        rnn_type=rng.choice(('gru', 'lstm')),  # rastgele tip
        hidden_size=rng.choice(GIZLI_BOYUTLAR),  # rastgele gizli boyut
        num_layers=rng.choice(KATMANLAR),  # rastgele katman
        bidirectional=rng.choice((True, False)),  # rastgele yön
        dropout=rng.choice(DROPOUTLAR),  # rastgele dropout
        pooling=rng.choice(HAVUZLAMALAR),  # rastgele havuzlama
        optim=OptimSettings(  # rastgele optimizasyon
            batch_size=rng.choice(BATCHLER),  # rastgele yığın
            learning_rate=round(lr, 5),  # log-düzgün lr
            weight_decay=rng.choice(WEIGHT_DECAYLER),  # rastgele weight decay
            patience=8,  # sabit sabır
        ),
    )
    return feature_cfg, model_cfg  # aday ikilisi


def _isci_kur(manifest_path: str, seed: int, cache_root: str, max_epochs: int) -> None:  # Her paralel işçi süreç başlarken BİR KEZ çalışır: bölmeyi kurar.

    global _FOLDS, _CACHE_ROOT, _MAX_EPOCHS  # süreç-genel durumu doldur
    import pandas as pd  # (fonksiyon içi import)
    from final.pipeline import SplitSettings
    from ser.data.splits import prepare_splits

    manifest = pd.read_csv(manifest_path)  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    train_df, val_df, _test = prepare_splits(manifest, ayar, seed=seed)  # böl (test kullanılmaz)
    _FOLDS = {'train': train_df, 'val': val_df}  # yalnız eğitim + geçerleme
    _CACHE_ROOT = cache_root  # önbellek kökünü sakla
    _MAX_EPOCHS = max_epochs  # epoch sınırını sakla


def _deneme_kos(girdi):  # Tek bir adayı eğitir; sonuç satırını döndürür (işçi süreçte koşar).

    k, feature_cfg, model_cfg = girdi  # deneme no + aday
    import torch  # (fonksiyon içi import)
    from final.dataset import Standardizer
    from final.features import extract_interval_series
    from final.models import SeqRNN, count_parameters
    from final.pipeline import _feature_folds
    from final.training import train_with_early_stopping
    from ser.constants import NUM_CLASSES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    # Öznitelikler ön-çıkarma sayesinde diskte hazır; buradaki yükleme saf okuma.
    tensorlar = _feature_folds(_FOLDS, feature_cfg, extract_interval_series,  # öznitelikleri yükle
                               _CACHE_ROOT, workers=8, cache=_FEATURE_CACHE)
    train_x, train_y = tensorlar['train']  # eğitim
    val_x, val_y = tensorlar['val']  # geçerleme
    olcek = Standardizer.fit(train_x, feature_axis=2)  # eğitimden normalizasyon
    model = SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)  # taze model
    basla = time.perf_counter()  # süre başlat
    sonuc = train_with_early_stopping(  # eğit
        model, olcek.transform(train_x), train_y,
        olcek.transform(val_x), val_y, model_cfg.optim,
        num_classes=NUM_CLASSES, device=device,
        max_epochs=_MAX_EPOCHS, seed=42,
    )
    saniye = time.perf_counter() - basla  # geçen süre
    m = sonuc.validation_metrics  # geçerleme metrikleri (test'e dokunma)
    return [k, feature_cfg.n_intervals, feature_cfg.interval_ms,  # CSV satırı
            model_cfg.rnn_type, model_cfg.hidden_size, model_cfg.num_layers,
            model_cfg.bidirectional, model_cfg.dropout, model_cfg.pooling,
            model_cfg.optim.batch_size, model_cfg.optim.learning_rate,
            model_cfg.optim.weight_decay, count_parameters(model),
            sonuc.best_epoch, round(m['accuracy'], 4),
            round(m['macro_f1'], 4), round(saniye, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--denemeler', type=int, default=60)  # kaç rastgele deneme
    parser.add_argument('--paralel', type=int, default=3,  # aynı anda kaç eğitim
                        help='Aynı anda kaç deneme eğitilsin (GPU paylaşımlı).')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')  # manifest
    parser.add_argument('--cache-root', default='data/cache/final')  # önbellek
    parser.add_argument('--out', default='final/deneyler/rastgele_arama')  # çıktı
    parser.add_argument('--seed', type=int, default=42)  # bölme tohumu
    parser.add_argument('--ornekleme-seed', type=int, default=1)  # örnekleme tohumu
    parser.add_argument('--max-epochs', type=int, default=60)  # en fazla epoch
    parser.add_argument('--feature-workers', type=int, default=24,  # ön-çıkarma işçisi
                        help='Ön-çıkarma iş parçacığı sayısı.')
    args = parser.parse_args()  # argümanları oku

    import pandas as pd  # (fonksiyon içi import)
    from final.features import extract_interval_series
    from final.pipeline import SplitSettings, _feature_folds
    from ser.data.splits import prepare_splits
    from ser.utils import ensure_dir, get_logger

    log = get_logger(__name__)  # günlükleyici

    # 1) Adayları baştan örnekle (hangi aralık düzenleri lazım, bilelim).
    rng = random.Random(args.ornekleme_seed)  # örnekleme üreteci
    adaylar = [rastgele_aday(rng) for _ in range(args.denemeler)]  # N rastgele aday

    # 2) ÖN-ÇIKARMA: gereken her aralık düzeninin öznitelikleri tek seferde.
    manifest = pd.read_csv(args.manifest)  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    train_df, val_df, _test = prepare_splits(manifest, ayar, seed=args.seed)  # böl
    folds = {'train': train_df, 'val': val_df}  # eğitim + geçerleme
    gereken = {f for f, _ in adaylar}  # gereken benzersiz aralık düzenleri
    log.info('Ön-çıkarma: %d farklı aralık düzeni, %d iş parçacığı',  # logla
             len(gereken), args.feature_workers)
    ge_basla = time.perf_counter()  # ön-çıkarma süresi
    for i, feature_cfg in enumerate(sorted(gereken,  # her düzen için
                                           key=lambda c: (c.n_intervals, c.interval_ms)), 1):
        _feature_folds(folds, feature_cfg, extract_interval_series,  # öznitelikleri önbelleğe yaz
                       args.cache_root, workers=args.feature_workers, cache={})
        log.info('  [%d/%d] %dx%dms hazır', i, len(gereken),  # ilerleme
                 feature_cfg.n_intervals, feature_cfg.interval_ms)
    log.info('Ön-çıkarma bitti: %.0f sn', time.perf_counter() - ge_basla)  # süreyi bildir

    # 3) PARALEL DENEME DÖNGÜSÜ.
    out_dir = ensure_dir(args.out)  # çıktı klasörü
    defter = Path(out_dir) / 'arama_defteri.csv'  # arama defteri yolu
    yeni_dosya = not defter.exists()  # dosya yeni mi (başlık gerekli mi)
    en_iyi_f1 = -1.0  # şimdiye kadarki en iyi macro-F1
    bitti = 0  # tamamlanan deneme sayısı
    tur_basla = time.perf_counter()  # döngü süresi
    with open(defter, 'a', newline='', encoding='utf-8') as f:  # deftere ekleme kipinde aç
        yazici = csv.writer(f)  # CSV yazıcı
        if yeni_dosya:  # yeni dosyaysa
            yazici.writerow(['deneme', 'n_intervals', 'interval_ms', 'rnn_type',  # başlık satırı
                             'hidden', 'layers', 'bidir', 'dropout', 'pooling',
                             'batch', 'lr', 'weight_decay', 'parametre',
                             'best_epoch', 'val_accuracy', 'val_macro_f1', 'saniye'])
        gorevler = [(k, fc, mc) for k, (fc, mc) in enumerate(adaylar, start=1)]  # numaralı işler
        with ProcessPoolExecutor(  # paralel süreç havuzu
            max_workers=args.paralel,  # aynı anda N eğitim
            initializer=_isci_kur,  # her işçide bölmeyi kur
            initargs=(args.manifest, args.seed, args.cache_root, args.max_epochs),
        ) as havuz:
            for satir in (fut.result() for fut in  # tamamlanan denemeleri topla
                          as_completed(havuz.submit(_deneme_kos, g) for g in gorevler)):
                yazici.writerow(satir)  # sonucu deftere yaz
                f.flush()  # hemen diske (çökse bile kayıp olmaz)
                bitti += 1  # sayacı artır
                f1 = satir[15]  # geçerleme macro-F1
                isaret = ''  # lider işareti
                if f1 > en_iyi_f1:  # yeni lider ise
                    en_iyi_f1 = f1  # lideri güncelle
                    isaret = '  <-- YENİ LİDER'  # işaretle
                log.info('[%d/%d] deneme#%d %dx%dms %s h%d L%d %s | val F1=%.4f%s',  # logla
                         bitti, len(gorevler), satir[0], satir[1], satir[2],
                         satir[3], satir[4], satir[5], satir[8], f1, isaret)
    log.info('Deneme döngüsü: %.0f sn (%d deneme, paralel=%d)',  # döngü süresini bildir
             time.perf_counter() - tur_basla, len(gorevler), args.paralel)

    # 4) Özet.
    tablo = pd.read_csv(defter).sort_values('val_macro_f1', ascending=False)  # deftere göre sırala
    print('\n===== EN İYİ 5 ADAY (geçerleme macro-F1) =====')  # başlık
    print(tablo.head(5).to_string(index=False))  # ilk 5 adayı yazdır
    resmi = 0.5638  # resmî kazananın geçerleme macro-F1'i
    lider = float(tablo.iloc[0]['val_macro_f1'])  # aramanın lideri
    if lider > resmi:  # lider resmîyi geçtiyse
        print(f'\nLider ({lider:.4f}) resmî kazananı ({resmi:.4f}) GEÇTİ — '
              'test teyidi için run_experiment ile bilinçli koşu yapın.')
    else:  # geçemediyse
        print(f'\nLider ({lider:.4f}) resmî kazananı ({resmi:.4f}) geçemedi — '
              'mevcut ayarlar savunuldu.')


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
