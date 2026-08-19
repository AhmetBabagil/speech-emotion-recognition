# İki karışıklık matrisini ayrı üretir:
# * test_confusion_matrix.png = taban arama-kazananı (winner_model.pt, 44 boyut) -> raporun "Yöntem 2 kazanan" bölümü kendi içinde tutarlı kalır.
# * son_confusion_matrix.png = nihai model (improved_model.pt, jitter+kontrast 53) -> sunum ve raporun nihai bölümü güncel modeli gösterir.
# Ayrıca her iki modelin test metriklerini yazdırır (tablo doldurmak için).

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # diziler
import pandas as pd  # noqa: E402  # manifest
import torch  # noqa: E402  # model yükleme

from final.ablasyon import MODEL, on_cikar_paralel  # noqa: E402  # model + paralel çıkarma
from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import IntervalConfig, extract_interval_series  # noqa: E402  # öznitelikler
from final.models import SeqRNN  # noqa: E402  # model
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402  # bölme + yükleme
from final.training import evaluate_arrays, inverse_frequency_weights  # noqa: E402  # değerlendirme + ağırlık
from ser.constants import NUM_CLASSES  # noqa: E402  # sınıf sayısı
from ser.data.splits import prepare_splits  # noqa: E402  # konuşmacı-bağımsız bölme
from ser.evaluate import compute_metrics, report as evaluate_report  # noqa: E402  # metrik + matris


def degerlendir(dosya, fcfg, folds, cw, device, out, prefix, baslik):  # kayıtlı bir modeli test'te değerlendirip matris çizer
    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),  # öznitelikleri paralel çıkar
                     dict(fcfg.__dict__), 'data/cache/final', 14)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',  # test tensörü
                       workers=8, cache={})
    ex, ey = T['test']  # test öznitelik + etiket
    d = torch.load(dosya, map_location='cpu', weights_only=False)  # kayıtlı modeli yükle
    olcek = Standardizer(mean=d['standardizer_mean'], scale=d['standardizer_scale'],  # kayıttaki normalizasyonu geri kur
                         feature_axis=d['feature_axis'])
    ex_s = olcek.transform(ex)  # test'i normalize et
    model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)  # model kabuğu
    model.load_state_dict(d['state_dict'])  # ağırlıkları yükle
    model.to(device).eval()  # cihaz + değerlendirme kipi
    _, _, prob = evaluate_arrays(model, ex_s, ey, class_weights=cw, device=device)  # test olasılıkları
    tm = evaluate_report(ey, prob.argmax(axis=1), out, prefix=prefix, title=baslik)  # metrik + matris yaz
    print(f'[{prefix}] {baslik}: boyut={fcfg.feature_dim} acc={tm["accuracy"]:.4f} '  # sonucu yazdır
          f'macro-F1={tm["macro_f1"]:.4f} disgustF1={tm["per_class"]["disgust"]["f1"]:.3f}')
    return tm  # metrikleri döndür


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları
    out = Path('final/outputs/cremad/rnn')  # çıktı klasörü

    # Taban arama-kazanani: tum oznitelik bayraklari kapali (44 boyut).
    taban = IntervalConfig(n_intervals=32, interval_ms=200,  # taban öznitelik seti
                           use_pitch=False, use_jitter=False, use_contrast=False,
                           use_delta2=False, use_bandwidth=False)
    degerlendir(out / 'winner_model.pt', taban, folds, cw, device, out,  # taban kazananın matrisi
                'test', 'Yontem 2 arama-kazanani (taban 44) test')

    # Nihai model: jitter+kontrast (53 boyut).
    nihai = IntervalConfig(n_intervals=32, interval_ms=200)  # nihai öznitelik seti (varsayılan)
    degerlendir(out / 'improved_model.pt', nihai, folds, cw, device, out,  # nihai modelin matrisi
                'son', 'Yontem 2 nihai (jitter+kontrast 53) test')


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
