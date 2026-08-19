# En yalın aday: SADECE spektral kontrast (pitch+jitter olmadan) = 51 boyut.
#
# Bırak-birini ablasyonu pitch ve jitter'ın kontrastla birlikteyken fazlalık olduğunu gösterdi. Bu betik "taban + yalnız kontrast" setini 5 koşu eğitip, bu en yalın kombinasyonun en iyisi olup olmadığını test eder.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # ortalama/std
import pandas as pd  # noqa: E402  # manifest
import torch  # noqa: E402  # cihaz + tohum

from final.ablasyon import MODEL, on_cikar_paralel  # noqa: E402  # kazanan model + paralel çıkarma
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


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    # SADECE kontrast: pitch=False, jitter=False, contrast=True.
    fcfg = IntervalConfig(n_intervals=32, interval_ms=200,  # yalnız kontrast öznitelik seti
                          use_pitch=False, use_jitter=False, use_contrast=True,
                          use_delta2=False, use_bandwidth=False)
    print(f'Sadece-kontrast set: boyut={fcfg.feature_dim}')  # boyutu bildir (51)

    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),  # öznitelikleri paralel çıkar
                     dict(fcfg.__dict__), 'data/cache/final', 14)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',  # katman tensörleri
                       workers=8, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']  # eğitim/geçerleme/test
    olcek = Standardizer.fit(tx, feature_axis=2)  # eğitimden normalizasyon
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et

    accs, f1s = [], []  # koşuların doğruluk/F1 sonuçları
    for k in range(5):  # 5 koşu (farklı tohum)
        torch.manual_seed(k)  # torch tohumu
        torch.cuda.manual_seed_all(k)  # GPU tohumu
        model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)  # taze model
        res = train_with_early_stopping(model, tx_s, ty, vx_s, vy, MODEL.optim,  # eğit
                                        num_classes=NUM_CLASSES, device=device,
                                        max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)  # test
        m = compute_metrics(ey, prob.argmax(axis=1))  # metrikler
        accs.append(m['accuracy'])  # doğruluğu biriktir
        f1s.append(m['macro_f1'])  # macro-F1'i biriktir

    print(f'\n===== SADECE KONTRAST ({fcfg.feature_dim} boyut) =====')  # başlık
    print(f'  test acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}')  # ortalama ± std
    print(f'  macro-F1 = {np.mean(f1s):.4f}')  # macro-F1 ortalaması
    print('\nKıyas (5 koşu ort.):')  # kıyaslar
    print('  taban (44)            = 0.6455')  # taban
    print('  pitch+jitter+kontrast (56) = 0.6636')  # tam set
    print('  - pitch (53)          = 0.6694')  # nihai (jitter+kontrast)
    print(f'  SADECE kontrast ({fcfg.feature_dim})    = {np.mean(accs):.4f}  <-- bu koşu')  # bu deneme


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
