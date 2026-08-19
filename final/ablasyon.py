# Öznitelik ablasyonu: her mantıklı öznitelik grubunun test başarısına katkısı.
#
# Kümülatif olarak öznitelik ekler (taban -> +pitch -> +jitter/shimmer -> +kontrast -> +delta2 -> +bant genişliği) ve her seti aynı resmî bölmede (seed 42) BİRDEN ÇOK kez eğitir. Neden çok koşu? GPU tam deterministik olmadığından tek koşu ±%1-2 oynayabilir; ortalama ± std gerçek katkıyı verir.
#
# Sonuç: rapora doğrudan girebilecek bir tablo + ablasyon.csv.
#
# Örnek: python final/ablasyon.py --kosu 5 --feature-workers 16

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
import csv  # sonuç tablosunu CSV'ye yazmak için
from pathlib import Path  # dosya yolları
import sys  # sys.path'e proje kökünü eklemek için

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # ortalama/std hesabı
import pandas as pd  # noqa: E402  # manifest okuma
import torch  # noqa: E402  # cihaz + tohum

from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import IntervalConfig, extract_interval_series  # noqa: E402  # aralık öznitelikleri
from final.models import OptimSettings, RNNConfig, SeqRNN  # noqa: E402  # model + ayarlar
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402  # bölme ayarı + öznitelik yükleme
from final.training import (  # noqa: E402  # eğitim + değerlendirme
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402  # sınıf sayısı (6)
from ser.data.splits import prepare_splits  # noqa: E402  # konuşmacı-bağımsız bölme
from ser.evaluate import compute_metrics  # noqa: E402  # metrik hesabı
from ser.utils import ensure_dir, get_logger  # noqa: E402  # klasör + günlük

log = get_logger(__name__)  # günlükleyici

# Kümülatif ablasyon setleri: her satır bir öncekine bir öznitelik grubu ekler.
SETLER = [
    ('taban',      dict(use_pitch=False, use_jitter=False, use_contrast=False, use_delta2=False, use_bandwidth=False)),  # yalnız taban 44
    ('+pitch',     dict(use_pitch=True,  use_jitter=False, use_contrast=False, use_delta2=False, use_bandwidth=False)),  # +perde
    ('+jit/shim',  dict(use_pitch=True,  use_jitter=True,  use_contrast=False, use_delta2=False, use_bandwidth=False)),  # +titreme
    ('+kontrast',  dict(use_pitch=True,  use_jitter=True,  use_contrast=True,  use_delta2=False, use_bandwidth=False)),  # +spektral kontrast
    ('+delta2',    dict(use_pitch=True,  use_jitter=True,  use_contrast=True,  use_delta2=True,  use_bandwidth=False)),  # +ivme
    ('+bant',      dict(use_pitch=True,  use_jitter=True,  use_contrast=True,  use_delta2=True,  use_bandwidth=True)),  # +bant genişliği
]

# Model: resmî kazananın ayarları — tüm setlerde AYNI, ki fark yalnız özniteliğe bağlı olsun.
MODEL = RNNConfig(rnn_type='gru', hidden_size=192, num_layers=2, bidirectional=True,  # kazanan BiGRU
                  dropout=0.3, pooling='mean',
                  optim=OptimSettings(batch_size=64, learning_rate=1e-3,
                                      weight_decay=1e-4, patience=8))


def on_cikar_paralel(yollar, cfg_sozluk, cache_root, islemler):
    # Bir öznitelik setini süreç-paralel çıkarır (hizli_cikarma altyapısını yeniden kullanır).
    #
    # pyin gibi pahalı işlemleri thread yerine 14 çekirdeğe böler; eksik olmayan dosyalar atlanır, böylece ikinci kez çağrıldığında saniyeler sürer.

    from concurrent.futures import ProcessPoolExecutor  # süreç havuzu
    from final.dataset import feature_cache_path  # önbellek yolu
    from final.features import IntervalConfig  # öznitelik ayarı
    from final.hizli_cikarma import _cikar, _isci_kur  # çıkarma işçisi + kurulum

    fp = IntervalConfig(**cfg_sozluk).fingerprint  # bu setin önbellek kimliği
    eksik = [(y, cfg_sozluk) for y in yollar  # önbellekte olmayan dosyalar
             if not feature_cache_path(y, cache_root, fp).is_file()]
    if not eksik:  # hepsi zaten çıkarılmışsa
        return  # dokunma
    log.info('  ön-çıkarma (paralel): %d dosya, %d süreç', len(eksik), islemler)  # ilerlemeyi yaz
    with ProcessPoolExecutor(max_workers=islemler,  # süreç havuzu (BLAS sabitli)
                             initializer=_isci_kur, initargs=(cache_root,)) as havuz:
        for _ in havuz.map(_cikar, eksik, chunksize=64):  # eksikleri paralel çıkar
            pass  # sonucu (yalnız yan etki: diske yazma) yoksay


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--kosu', type=int, default=5,  # her set kaç kez eğitilsin
                        help='Her set kaç kez eğitilsin (ortalama için).')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')  # manifest yolu
    parser.add_argument('--cache-root', default='data/cache/final')  # önbellek kökü
    parser.add_argument('--out', default='final/deneyler/ablasyon')  # çıktı klasörü
    parser.add_argument('--feature-workers', type=int, default=8)  # okuma işçisi
    parser.add_argument('--islemler', type=int, default=14,  # ön-çıkarma süreç sayısı
                        help='Ön-çıkarma için paralel süreç sayısı.')
    parser.add_argument('--max-epochs', type=int, default=60)  # en fazla epoch
    args = parser.parse_args()  # argümanları oku

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    manifest = pd.read_csv(args.manifest)  # manifesti oku
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    out_dir = ensure_dir(args.out)  # çıktı klasörünü oluştur
    defter = Path(out_dir) / 'ablasyon.csv'  # sonuç CSV yolu
    satirlar = []  # tablo satırları
    taban_acc = None  # taban doğruluğu (katkıyı buna göre ölç)

    tum_yollar = pd.concat([tr, va, te])['path'].astype(str).tolist()  # tüm dosya yolları

    for ad, bayrak in SETLER:  # her ablasyon seti için
        fcfg = IntervalConfig(n_intervals=32, interval_ms=200, **bayrak)  # bu setin öznitelik ayarı
        # ÖN-ÇIKARMA (süreç-paralel): bu setin öznitelikleri önbellekte yoksa,
        # 14 çekirdekle hızlıca doldur. Böylece aşağıdaki thread'li okuma saf
        # önbellek okuması olur, pyin beklemesi olmaz.
        on_cikar_paralel(tum_yollar, dict(fcfg.__dict__), args.cache_root, args.islemler)  # önce paralel çıkar
        # Öznitelikler artık hazır -> hızlı okuma.
        T = _feature_folds(folds, fcfg, extract_interval_series, args.cache_root,  # katman tensörleri
                           workers=args.feature_workers, cache={})
        tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']  # eğitim/geçerleme/test
        olcek = Standardizer.fit(tx, feature_axis=2)  # eğitimden normalizasyon öğren
        tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et

        accs, f1s = [], []  # koşuların doğruluk/F1 sonuçları
        for k in range(args.kosu):  # her koşu için (farklı tohum)
            torch.manual_seed(k)  # torch tohumu
            torch.cuda.manual_seed_all(k)  # GPU tohumu
            model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)  # taze BiGRU
            res = train_with_early_stopping(  # eğit
                model, tx_s, ty, vx_s, vy, MODEL.optim,
                num_classes=NUM_CLASSES, device=device,
                max_epochs=args.max_epochs, seed=k)
            _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)  # test'te değerlendir
            m = compute_metrics(ey, prob.argmax(axis=1))  # metrikler
            accs.append(m['accuracy'])  # doğruluğu biriktir
            f1s.append(m['macro_f1'])  # macro-F1'i biriktir

        acc_ort, acc_std = float(np.mean(accs)), float(np.std(accs))  # doğruluk ort ± std
        f1_ort = float(np.mean(f1s))  # macro-F1 ortalaması
        if taban_acc is None:  # ilk set (taban) ise
            taban_acc = acc_ort  # tabanı kaydet
        katki = acc_ort - taban_acc  # tabana göre katkı
        satirlar.append({  # tablo satırı
            'set': ad, 'boyut': fcfg.feature_dim,
            'test_acc_ort': round(acc_ort, 4), 'test_acc_std': round(acc_std, 4),
            'test_f1_ort': round(f1_ort, 4),
            'tabana_katki': round(katki, 4),
        })
        log.info('[%s] boyut=%d | test acc=%.4f±%.4f  macro-F1=%.4f | tabana katkı %+.4f',  # logla
                 ad, fcfg.feature_dim, acc_ort, acc_std, f1_ort, katki)

    # Tabloyu yaz + ekrana bas.
    with open(defter, 'w', newline='', encoding='utf-8') as f:  # CSV dosyası
        yazici = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))  # sütun başlıkları
        yazici.writeheader()  # başlık satırı
        yazici.writerows(satirlar)  # veri satırları

    print(f'\n===== ÖZNİTELİK ABLASYONU ({args.kosu} koşu ortalaması, seed 42) =====')  # başlık
    print(f'{"set":12s} {"boyut":>5s} {"test acc":>12s} {"macro-F1":>9s} {"katkı":>8s}')  # sütun adları
    for s in satirlar:  # her satırı yazdır
        print(f'{s["set"]:12s} {s["boyut"]:5d} '
              f'{s["test_acc_ort"]:.4f}±{s["test_acc_std"]:.4f} '
              f'{s["test_f1_ort"]:8.4f} {s["tabana_katki"]:+8.4f}')
    print(f'\nDefter: {defter}')  # CSV yolunu bildir


if __name__ == '__main__':  # betik doğrudan çalıştırılırsa
    main()  # ana fonksiyonu koştur
