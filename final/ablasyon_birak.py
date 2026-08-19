# Bırak-birini (leave-one-out) ablasyonu: her öznitelik grubunun GERÇEKTEN gerekli olup olmadığını ölçer.
#
# Kümülatif ablasyon "sırayla eklersem ne olur" der; bu ise tam setten her grubu TEK TEK çıkarıp "onsuz ne kadar kötüleşiyorum" diye bakar. Bir grubu çıkarınca doğruluk düşüyorsa o grup gerekli; değişmiyor/artıyorsa gereksiz.
#
# Tam set: MFCC(39) + skaler(5) + pitch(3) + jitter(2) + kontrast(7) = 56. (delta2 ve bant kümülatif ablasyonda zaten düştüğü için dahil değil.)
#
# Örnek: python final/ablasyon_birak.py --kosu 5 --islemler 14

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
from pathlib import Path  # dosya yolları
import sys  # import yolu için

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # ortalama/std
import pandas as pd  # noqa: E402  # manifest
import torch  # noqa: E402  # cihaz + tohum

from final.ablasyon import MODEL, on_cikar_paralel  # noqa: E402  # kazanan model + paralel çıkarma (yeniden kullan)
from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import IntervalConfig, extract_interval_series  # noqa: E402  # öznitelikler
from final.models import SeqRNN  # noqa: E402  # model
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402  # bölme + yükleme
from final.training import (  # noqa: E402  # eğitim + değerlendirme
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402  # sınıf sayısı
from ser.data.splits import prepare_splits  # noqa: E402  # konuşmacı-bağımsız bölme
from ser.evaluate import compute_metrics  # noqa: E402  # metrikler
from ser.utils import get_logger  # noqa: E402  # günlük

log = get_logger(__name__)  # günlükleyici

# Tam set = pitch + jitter + kontrast açık (delta2/bant kapalı — zaten kötüydü).
# Her deneme, tam setten YALNIZ birini kapatır.
TAM = dict(use_pitch=True, use_jitter=True, use_contrast=True,  # tam öznitelik seti (56 boyut)
           use_delta2=False, use_bandwidth=False)
GRUPLAR = [
    ('TAM SET (56)', TAM),  # referans: hepsi açık
    ('- pitch',    {**TAM, 'use_pitch': False}),  # pitch'i çıkar
    ('- jitter',   {**TAM, 'use_jitter': False}),  # jitter'ı çıkar
    ('- kontrast', {**TAM, 'use_contrast': False}),  # kontrastı çıkar
]


def egit_test(fcfg, folds, cw, device, kosu, workers, islemler):  # bir öznitelik setini N kez eğitip test doğruluğunu döndürür
    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),  # önce özniteliği paralel çıkar
                     dict(fcfg.__dict__), 'data/cache/final', islemler)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',  # katman tensörleri
                       workers=workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']  # eğitim/geçerleme/test
    olcek = Standardizer.fit(tx, feature_axis=2)  # eğitimden normalizasyon öğren
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et
    accs = []  # koşuların doğrulukları
    for k in range(kosu):  # her koşu (farklı tohum)
        torch.manual_seed(k)  # torch tohumu
        torch.cuda.manual_seed_all(k)  # GPU tohumu
        model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)  # taze model
        res = train_with_early_stopping(model, tx_s, ty, vx_s, vy, MODEL.optim,  # eğit
                                        num_classes=NUM_CLASSES, device=device,
                                        max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)  # test
        accs.append(compute_metrics(ey, prob.argmax(axis=1))['accuracy'])  # doğruluğu biriktir
    return float(np.mean(accs)), float(np.std(accs))  # ortalama ± std


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kosu', type=int, default=5)  # koşu sayısı
    parser.add_argument('--feature-workers', type=int, default=8)  # okuma işçisi
    parser.add_argument('--islemler', type=int, default=14)  # çıkarma süreci
    args = parser.parse_args()  # argümanları oku

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    tam_acc = None  # tam setin doğruluğu (etkiyi buna göre ölç)
    satirlar = []  # sonuç satırları
    for ad, bayrak in GRUPLAR:  # her grup için
        fcfg = IntervalConfig(n_intervals=32, interval_ms=200, **bayrak)  # bu grubun öznitelik ayarı
        acc, std = egit_test(fcfg, folds, cw, device, args.kosu,  # eğit + test
                             args.feature_workers, args.islemler)
        if tam_acc is None:  # ilk (TAM SET) ise
            tam_acc = acc  # referansı kaydet
        etki = acc - tam_acc   # tam sete göre: negatifse "o grup gerekliydi"
        satirlar.append((ad, fcfg.feature_dim, acc, std, etki))  # satırı ekle
        log.info('[%s] boyut=%d | acc=%.4f±%.4f | tam sete göre %+.4f',  # logla
                 ad, fcfg.feature_dim, acc, std, etki)

    print('\n===== BIRAK-BİRİNİ ABLASYONU (5 koşu ort.) =====')  # başlık
    print(f'{"grup":14s} {"boyut":>5s} {"test acc":>14s} {"etki":>9s}')  # sütun adları
    for ad, boyut, acc, std, etki in satirlar:  # her satır
        yorum = ''  # gerekli/gereksiz notu
        if ad != 'TAM SET (56)':  # referans satırı değilse
            yorum = '  <- gerekli' if etki < -0.003 else ('  <- gereksiz' if etki > -0.003 else '')  # yorumu belirle
        print(f'{ad:14s} {boyut:5d} {acc:.4f}±{std:.4f} {etki:+8.4f}{yorum}')  # satırı yazdır
    print('\nYorum: bir grubu ÇIKARINCA doğruluk çok düşüyorsa (negatif etki) o grup')  # açıklama
    print('gereklidir; değişmiyor/artıyorsa çıkarılabilir (gereksiz).')  # açıklama


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
