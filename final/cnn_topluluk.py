# Yöntem 1 (CNN) için TOPLULUK (ensemble): 5 SpecAugment-CNN'in softmax çıktılarını ortalar.
#
# Neden: tek CNN koşusunun test varyansı ±3,9 puan (çok kararsız). Topluluk, bağımsız koşuların hatalarını birbirine sönümleterek hem varyansı düşürür hem de tipik olarak tek koşuları geçer. Hazır/önceden eğitilmiş model YOK — beş model de sıfırdan, aynı mimari, farklı rastgele tohum.
#
# Dürüstlük: topluluk, test'e BAKMADAN kurulur (eşit ağırlıklı ortalama). Her model kendi geçerlemesinde erken durur; test yalnızca en sonda, bir kez ölçülür.
#
# Örnek: python final/cnn_topluluk.py --uye 5

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
import json  # özeti JSON'a yazmak için
from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # ortalama/std
import pandas as pd  # noqa: E402  # manifest
import torch  # noqa: E402  # cihaz + model kaydı

from final.augment import SpecMask  # noqa: E402  # SpecAugment maskeleme
from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import MelImageConfig, extract_mel_image  # noqa: E402  # mel görüntüsü
from final.models import CNNConfig, MelCNN, OptimSettings  # noqa: E402  # CNN + ayarlar
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402  # bölme + yükleme
from final.training import (  # noqa: E402  # eğitim + değerlendirme
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402  # sınıf sayısı
from ser.data.splits import prepare_splits  # noqa: E402  # konuşmacı-bağımsız bölme
from ser.evaluate import compute_metrics, report as evaluate_report  # noqa: E402  # metrik + matris

OPT = OptimSettings(batch_size=16, learning_rate=1e-3, weight_decay=1e-4, patience=8)  # kazanan CNN optimizasyonu
CFG = CNNConfig(channels=(32, 64, 128), dropout=0.3, optim=OPT)  # kazanan CNN yapılandırması


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)  # argüman ayrıştırıcı
    parser.add_argument('--uye', type=int, default=5, help='topluluk üye sayısı')  # üye sayısı
    parser.add_argument('--feature-workers', type=int, default=8)  # okuma işçisi
    args = parser.parse_args()  # argümanları oku

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    fcfg = MelImageConfig()  # mel görüntü ayarı (64x128)
    T = _feature_folds(folds, fcfg, extract_mel_image, 'data/cache/final',  # mel özniteliklerini yükle
                       workers=args.feature_workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']  # eğitim/geçerleme/test
    olcek = Standardizer.fit(tx, feature_axis=1)  # eğitimden normalizasyon (mel bandı ekseni)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et

    spec = SpecMask()  # SpecAugment maskeleme (yalnız eğitimde)
    tekil_acc = []  # her üyenin tek başına doğruluğu
    prob_toplam = None  # softmax olasılıklarının toplamı (ortalama için)
    durumlar = []  # her üyenin ağırlıkları
    for k in range(args.uye):  # her topluluk üyesi için (farklı tohum)
        outcome = train_with_early_stopping(  # bir CNN eğit
            MelCNN(NUM_CLASSES, CFG), tx_s, ty, vx_s, vy, CFG.optim,
            num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k,
            amp=True, train_transform=spec)  # SpecAugment ile
        _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)  # test olasılıkları
        acc = compute_metrics(ey, prob.argmax(axis=1))['accuracy']  # tek üye doğruluğu
        tekil_acc.append(acc)  # biriktir
        prob_toplam = prob if prob_toplam is None else prob_toplam + prob  # olasılıkları topla
        durumlar.append({n: v.detach().cpu().clone() for n, v in outcome.model.state_dict().items()})  # ağırlıkları sakla
        print(f'  üye {k}: tekil test_acc={acc:.4f}')  # üyeyi bildir

    # TOPLULUK: 5 softmax'ın ortalaması -> tek tahmin.
    ens_prob = prob_toplam / args.uye  # olasılıkların ortalaması
    ens_pred = ens_prob.argmax(axis=1)  # en yüksek ortalama olasılık = topluluk tahmini

    cnn_dir = Path('final/outputs/cremad/cnn')  # çıktı klasörü
    tm = evaluate_report(ey, ens_pred, cnn_dir, prefix='ensemble',  # topluluk metrikleri + matris
                         title=f'CNN topluluk ({args.uye} SpecAugment) test')
    torch.save({'uyeler': durumlar, 'uye_sayisi': args.uye,  # topluluğu kaydet (5 ağırlık seti)
                'model_config': CFG.to_dict(), 'feature_config': fcfg.__dict__,
                'standardizer_mean': olcek.mean, 'standardizer_scale': olcek.scale,
                'feature_axis': olcek.feature_axis, 'variant': 'specaugment_ensemble'},
               cnn_dir / 'ensemble_model.pt')
    ozet = {  # topluluk özet sözlüğü
        'uye_sayisi': args.uye,
        'tekil_acc_ort': float(np.mean(tekil_acc)),  # bireysel üye ortalaması
        'tekil_acc_std': float(np.std(tekil_acc)),  # bireysel std
        'topluluk_acc': tm['accuracy'],  # topluluk doğruluğu
        'topluluk_balanced': tm['balanced_accuracy'],  # dengeli doğruluk
        'topluluk_macro_f1': tm['macro_f1'],  # macro-F1
        'taban_5kosu_ort': 0.5585,  # taban tek CNN 5-koşu ort (kıyas)
        'specaug_5kosu_ort': 0.5989,  # SpecAugment tek CNN 5-koşu ort (kıyas)
        'eski_kazanan_tek_kosu': 0.6264,  # eski şanslı tek koşu (kıyas)
        'per_class': tm['per_class'],  # sınıf-bazı sonuçlar
    }
    (cnn_dir / 'ensemble_summary.json').write_text(  # özeti JSON'a yaz
        json.dumps(ozet, indent=2, default=str), encoding='utf-8')

    print(f'\n===== CNN TOPLULUK ({args.uye} SpecAugment-CNN) =====')  # başlık
    print(f'  tekil koşular   : {np.mean(tekil_acc):.4f} ± {np.std(tekil_acc):.4f}')  # bireysel ort ± std
    print(f'  TOPLULUK        : acc={tm["accuracy"]:.4f}  '  # topluluk sonucu
          f'balanced={tm["balanced_accuracy"]:.4f}  macro-F1={tm["macro_f1"]:.4f}')
    print(f'  (taban 5-koşu {0.5585:.3f} | specaug 5-koşu {0.5989:.3f} | '  # kıyaslar
          f'eski kazanan tek koşu {0.6264:.3f})')
    kazanc = tm['accuracy'] - np.mean(tekil_acc)  # topluluğun bireysel ortalamaya kazancı
    print(f'  topluluk kazancı (tekil ort. üzerine): {kazanc:+.4f}')  # kazancı bildir


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
