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

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
from dataclasses import replace  # ayarın bir alanını değiştirip kopyalamak için
from pathlib import Path  # dosya yolları
import sys  # import yolu
import time  # süre ölçümü

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # ortalama/std
import pandas as pd  # noqa: E402  # manifest
import torch  # noqa: E402  # cihaz

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
from ser.evaluate import compute_metrics  # noqa: E402  # metrikler

# Kazanan CNN yapılandırması (winner.json ile aynı).
OPT = OptimSettings(batch_size=16, learning_rate=1e-3, weight_decay=1e-4, patience=8)  # kazanan optimizasyon
TABAN_CFG = CNNConfig(channels=(32, 64, 128), dropout=0.3, optim=OPT)  # kazanan CNN ayarı


def varyantlar():  # (ad, model_cfg, transform, label_smoothing, mixup_alpha) listesi.
    return [
        ('taban',         TABAN_CFG, None, 0.0, 0.0),  # artırma yok (dürüst taban)
        ('specaug',       TABAN_CFG, SpecMask(), 0.0, 0.0),  # SpecAugment
        ('specaug_guclu', TABAN_CFG,  # güçlü SpecAugment
         SpecMask(freq_masks=2, freq_width=12, time_masks=2, time_width=24), 0.0, 0.0),
        ('label_smooth',  TABAN_CFG, None, 0.1, 0.0),  # etiket yumuşatma
        ('mixup',         TABAN_CFG, None, 0.0, 0.2),  # mixup
        ('specaug_ls',    TABAN_CFG, SpecMask(), 0.1, 0.0),  # specaug + label smoothing
        ('derin',         replace(TABAN_CFG, channels=(32, 64, 128, 256)), SpecMask(), 0.0, 0.0),  # 4 blok + specaug
        ('genis_drop',    replace(TABAN_CFG, channels=(48, 96, 192), dropout=0.4),  # geniş + yüksek dropout
         SpecMask(), 0.0, 0.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kosu', type=int, default=5)  # her varyant kaç koşu
    parser.add_argument('--sadece', nargs='+', default=None,  # yalnız bu varyantlar
                        help='yalnız bu varyant adlarını koştur')
    parser.add_argument('--feature-workers', type=int, default=8)  # okuma işçisi
    args = parser.parse_args()  # argümanları oku

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    folds = {'train': tr, 'val': va, 'test': te}  # katmanlar
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    # Mel öznitelikleri (önbellekten anında gelir; feature_axis=1).
    fcfg = MelImageConfig()  # mel görüntü ayarı
    T = _feature_folds(folds, fcfg, extract_mel_image, 'data/cache/final',  # mel özniteliklerini yükle
                       workers=args.feature_workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']  # eğitim/geçerleme/test
    olcek = Standardizer.fit(tx, feature_axis=1)  # eğitimden normalizasyon
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et
    print(f'Mel şekli: train={tx_s.shape}  (n_mels×kare)')  # boyutu bildir

    secili = varyantlar()  # tüm varyantlar
    if args.sadece:  # --sadece verildiyse
        secili = [v for v in secili if v[0] in set(args.sadece)]  # yalnız istenenler

    satirlar = []  # sonuç satırları
    for ad, cfg, tf, ls, mix in secili:  # her varyant için
        vfs, accs, f1s = [], [], []  # geçerleme F1, test acc, test F1
        t0 = time.perf_counter()  # süre başlat
        for k in range(args.kosu):  # her koşu (farklı tohum)
            outcome = train_with_early_stopping(  # varyantı eğit
                MelCNN(NUM_CLASSES, cfg), tx_s, ty, vx_s, vy, cfg.optim,
                num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k,
                amp=True, train_transform=tf, label_smoothing=ls, mixup_alpha=mix)  # tekniği uygula
            vfs.append(outcome.validation_metrics['macro_f1'])  # geçerleme F1 (seçim ölçütü)
            _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)  # test
            m = compute_metrics(ey, prob.argmax(axis=1))  # metrikler
            accs.append(m['accuracy']); f1s.append(m['macro_f1'])  # test sonuçlarını biriktir
        sn = (time.perf_counter() - t0) / max(args.kosu, 1)  # koşu başına süre
        satirlar.append((ad, float(np.mean(vfs)), float(np.mean(accs)),  # varyant sonucu
                         float(np.std(accs)), float(np.mean(f1s)), sn))
        print(f'  [{ad:14s}] val_macroF1={np.mean(vfs):.4f}  '  # varyantı bildir
              f'test_acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
              f'test_F1={np.mean(f1s):.4f}  ({sn:.0f} sn/koşu)')

    # Sonuçları JSON'a yaz (arka plan koşusunu sonradan okumak için).
    import json  # JSON yazma
    kayit = [{'varyant': ad, 'val_macro_f1': vf, 'test_acc': acc,  # sonuçları sözlüğe çevir
              'test_acc_std': std, 'test_macro_f1': f1, 'sn_kosu': sn}
             for ad, vf, acc, std, f1, sn in satirlar]
    Path('final/outputs/cremad/cnn').mkdir(parents=True, exist_ok=True)  # klasörü oluştur
    Path('final/outputs/cremad/cnn/gelistirme_deney.json').write_text(  # JSON'a yaz
        json.dumps({'kosu': args.kosu, 'sonuclar': kayit}, indent=2), encoding='utf-8')

    # Geçerleme macro-F1'e göre sırala (dürüst seçim ölçütü).
    satirlar.sort(key=lambda r: r[1], reverse=True)  # geçerleme F1'e göre azalan
    print('\n===== CNN GELİŞTİRME (5 koşu ort., geçerlemeye göre sıralı) =====')  # başlık
    print(f'{"varyant":15s} {"val macroF1":>12s} {"test acc":>14s} {"test F1":>9s}')  # sütun adları
    for ad, vf, acc, std, f1, _sn in satirlar:  # her satır
        yildiz = '  <-- en iyi (val)' if (ad, vf) == (satirlar[0][0], satirlar[0][1]) else ''  # lider işareti
        print(f'{ad:15s} {vf:12.4f} {acc:.4f}±{std:.4f} {f1:8.4f}{yildiz}')  # satırı yazdır
    print('\nSeçim geçerlemeye göre; test yalnız bilgi. Taban ile kıyasla.')  # not


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
