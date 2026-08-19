# Yöntem 2 (BiGRU) için TOPLULUK: 5 jitter+kontrast BiGRU'nun softmax çıktılarını ortalar.
#
# CNN'de topluluk +2,5 puan kazandırdı; aynı fikri en iyi modelimize (5-koşu ort. %66,9) uyguluyoruz. Beş üye de aynı mimari (BiGRU 32×200 ms, 192×2, jitter+kontrast 53 boyut), farklı rastgele tohum, hepsi sıfırdan.
#
# Dürüstlük: topluluk test'e BAKMADAN kurulur (eşit ağırlıklı ortalama). Her üye kendi geçerlemesinde erken durur; test yalnızca en sonda bir kez ölçülür.
#
# Örnek: python final/rnn_topluluk.py --uye 5

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
import json  # özeti JSON'a yazmak için
from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # ortalama/std
import pandas as pd  # noqa: E402  # manifest
import torch  # noqa: E402  # cihaz + model kaydı

from final.ablasyon import MODEL, on_cikar_paralel  # noqa: E402  # kazanan BiGRU + paralel çıkarma
from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import IntervalConfig, extract_interval_series  # noqa: E402  # aralık öznitelikleri
from final.models import SeqRNN  # noqa: E402  # model
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402  # bölme + yükleme
from final.training import (  # noqa: E402  # eğitim + değerlendirme
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402  # sınıf sayısı
from ser.data.splits import prepare_splits  # noqa: E402  # konuşmacı-bağımsız bölme
from ser.evaluate import compute_metrics, report as evaluate_report  # noqa: E402  # metrik + matris


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)  # argüman ayrıştırıcı
    parser.add_argument('--uye', type=int, default=5)  # topluluk üye sayısı
    parser.add_argument('--islemler', type=int, default=14)  # ön-çıkarma süreci
    parser.add_argument('--feature-workers', type=int, default=8)  # okuma işçisi
    args = parser.parse_args()  # argümanları oku

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    fcfg = IntervalConfig(n_intervals=32, interval_ms=200)  # jitter+kontrast (53)
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),  # öznitelikleri paralel çıkar
                     dict(fcfg.__dict__), 'data/cache/final', args.islemler)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',  # katman tensörleri
                       workers=args.feature_workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']  # eğitim/geçerleme/test
    olcek = Standardizer.fit(tx, feature_axis=2)  # eğitimden normalizasyon (öznitelik ekseni)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et

    tekil_acc = []  # her üyenin tek başına doğruluğu
    prob_toplam = None  # softmax olasılık toplamı
    durumlar = []  # her üyenin ağırlıkları
    for k in range(args.uye):  # her topluluk üyesi (farklı tohum)
        torch.manual_seed(k)  # torch tohumu
        torch.cuda.manual_seed_all(k)  # GPU tohumu
        outcome = train_with_early_stopping(  # bir BiGRU eğit
            SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL), tx_s, ty, vx_s, vy,
            MODEL.optim, num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)  # test olasılıkları
        acc = compute_metrics(ey, prob.argmax(axis=1))['accuracy']  # tek üye doğruluğu
        tekil_acc.append(acc)  # biriktir
        prob_toplam = prob if prob_toplam is None else prob_toplam + prob  # olasılıkları topla
        durumlar.append({n: v.detach().cpu().clone() for n, v in outcome.model.state_dict().items()})  # ağırlıkları sakla
        print(f'  üye {k}: tekil test_acc={acc:.4f}')  # üyeyi bildir

    ens_prob = prob_toplam / args.uye  # olasılıkların ortalaması
    ens_pred = ens_prob.argmax(axis=1)  # topluluk tahmini

    rnn_dir = Path('final/outputs/cremad/rnn')  # çıktı klasörü
    tm = evaluate_report(ey, ens_pred, rnn_dir, prefix='ensemble',  # topluluk metrikleri + matris
                         title=f'BiGRU topluluk ({args.uye} jitter+kontrast) test')
    torch.save({'uyeler': durumlar, 'uye_sayisi': args.uye,  # topluluğu kaydet (5 ağırlık seti)
                'model_config': MODEL.to_dict(), 'feature_config': fcfg.__dict__,
                'standardizer_mean': olcek.mean, 'standardizer_scale': olcek.scale,
                'feature_axis': olcek.feature_axis, 'variant': 'jitter_contrast_ensemble'},
               rnn_dir / 'ensemble_model.pt')
    ozet = {  # topluluk özet sözlüğü
        'uye_sayisi': args.uye,
        'tekil_acc_ort': float(np.mean(tekil_acc)),  # bireysel üye ortalaması
        'tekil_acc_std': float(np.std(tekil_acc)),  # bireysel std
        'topluluk_acc': tm['accuracy'],  # topluluk doğruluğu
        'topluluk_balanced': tm['balanced_accuracy'],  # dengeli doğruluk
        'topluluk_macro_f1': tm['macro_f1'],  # macro-F1
        'tekil_5kosu_ort': 0.6694,  # tek model 5-koşu ort (kıyas)
        'per_class': tm['per_class'],  # sınıf-bazı sonuçlar
    }
    (rnn_dir / 'ensemble_summary.json').write_text(  # özeti JSON'a yaz
        json.dumps(ozet, indent=2, default=str), encoding='utf-8')

    print(f'\n===== BiGRU TOPLULUK ({args.uye} jitter+kontrast-BiGRU) =====')  # başlık
    print(f'  tekil koşular : {np.mean(tekil_acc):.4f} ± {np.std(tekil_acc):.4f}')  # bireysel ort ± std
    print(f'  TOPLULUK      : acc={tm["accuracy"]:.4f}  '  # topluluk sonucu
          f'balanced={tm["balanced_accuracy"]:.4f}  macro-F1={tm["macro_f1"]:.4f}')
    print(f'  (tek model 5-koşu ort. {0.6694:.3f})')  # kıyas
    print(f'  topluluk kazancı: {tm["accuracy"] - np.mean(tekil_acc):+.4f}')  # kazancı bildir


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
