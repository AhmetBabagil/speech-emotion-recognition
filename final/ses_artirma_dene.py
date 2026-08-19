# Ses-uzayı (dalga formu) veri artırma denemesi — Yöntem 2 (BiGRU) için.
#
# Şimdiye kadarki tüm artırmalar spektrogram/öznitelik uzayındaydı (yalnızca gizleme). Bu deney dalga formunun KENDİSİNİ çeşitlendirir: her eğitim kaydının pitch-shift edilmiş ve hafif gürültü eklenmiş kopyalarını üreterek eğitim setine gerçekten yeni varyasyon ekler.
#
# Dürüst protokol: artırma YALNIZCA eğitim kayıtlarına uygulanır; geçerleme ve test orijinal hâlleriyle kalır. Hazır/önceden eğitilmiş model yok — sadece librosa ile sinyal işleme (yönerge izinli).
#
# ÖNEMLİ (Windows): torch import'ları main() İÇİNE ertelenmiştir. Böylece ProcessPoolExecutor işçileri (spawn ile bu modülü yeniden import eder) torch/CUDA yüklemez; yoksa işçiler CUDA çakışmasından takılıyor. Modül-üstü importlar sadece numpy + librosa + features (hafif) olmalı.
#
# Örnek: python final/ses_artirma_dene.py --kosu 5 --turler pitch noise --islemler 14

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

# BLAS/numba iş parçacıklarını numpy/librosa import'undan ÖNCE 1'e sabitle.
import os  # ortam değişkenleri (iş parçacığı sabitleme)

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):  # her BLAS değişkeni için
    os.environ.setdefault(_v, '1')  # 1'e sabitle (aşırı-abonelik/deadlock önle)

import argparse  # komut satırı argümanları
from concurrent.futures import ProcessPoolExecutor  # süreç havuzu
from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

# HAFİF importlar (işçiler bunları yükler; torch YOK):
import numpy as np  # noqa: E402  # diziler
from final.features import IntervalConfig, extract_interval_series, _load_audio  # noqa: E402  # öznitelik + ses yükleme


def _isci_kur() -> None:  # Her işçide BLAS iş parçacıklarını 1'e sabitle.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):  # her değişken
        os.environ[var] = '1'  # 1'e sabitle


def _dalga_artir(audio: np.ndarray, sr: int, tur: str, seed: int) -> np.ndarray:  # Dalga formuna tek bir artırma uygular (deterministik, seed'e bağlı).
    import librosa  # (işçi içinde) ses işleme
    rng = np.random.default_rng(seed)  # deterministik rastgele üreteç
    if tur == 'pitch':  # perde kaydırma
        adim = float(rng.choice([-3, -2, -1, 1, 2, 3]))   # ±1..3 yarım ton
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=adim)  # perdeyi kaydır
    if tur == 'noise':  # gürültü ekleme
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8  # sinyal enerjisi
        snr_db = float(rng.uniform(15, 30))               # 15-30 dB SNR
        gurultu_rms = rms / (10 ** (snr_db / 20))  # hedef SNR için gürültü enerjisi
        return audio + rng.normal(0, gurultu_rms, size=audio.shape).astype(audio.dtype)  # gürültü ekle
    return audio  # bilinmeyen tür -> dokunma


def _isci(arg):  # (yol, etiket, cfg_dict, tur, seed) -> (seri, etiket). torch kullanmaz.
    yol, etiket, cfg_dict, tur, seed = arg  # işi aç
    cfg = IntervalConfig(**cfg_dict)  # ayar nesnesi
    audio = _load_audio(yol, cfg.sample_rate)  # sesi yükle
    if tur != 'orig':  # orijinal değilse
        audio = _dalga_artir(audio, cfg.sample_rate, tur, seed).astype(np.float32)  # artır
    seri = extract_interval_series(yol, cfg, audio=audio)  # (artırılmış) sesten öznitelik çıkar
    return seri, etiket  # seri + etiket


def artirilmis_egitim(train_df, cfg, turler, islemler):  # Orijinal + artırılmış eğitim serilerini paralel üretir -> (X, y).
    isler = []  # (yol, etiket, cfg, tür, seed) işleri
    for i, row in enumerate(train_df.itertuples()):  # her eğitim kaydı için
        yol = str(row.path); etiket = int(row.label_idx); cd = dict(cfg.__dict__)  # yol/etiket/ayar
        isler.append((yol, etiket, cd, 'orig', 0))  # orijinal versiyon
        for j, tur in enumerate(turler):  # her artırma türü için
            isler.append((yol, etiket, cd, tur, i * 31 + j * 7 + 1))  # artırılmış versiyon (deterministik seed)
    X, y = [], []  # öznitelikler + etiketler
    if islemler <= 1:  # tek süreç istendiyse
        # Tek-süreç: numba+multiprocessing deadlock'unu tamamen atlar (yavaş ama sağlam).
        for k, is_ in enumerate(isler):  # işleri sırayla işle
            seri, etiket = _isci(is_)  # özniteliği çıkar
            X.append(seri); y.append(etiket)  # biriktir
            if k % 1000 == 0:  # her 1000'de bir
                print(f'    ...{k}/{len(isler)} çıkarıldı', flush=True)  # ilerleme
    else:  # paralel
        with ProcessPoolExecutor(max_workers=islemler, initializer=_isci_kur) as ex:  # süreç havuzu
            for seri, etiket in ex.map(_isci, isler, chunksize=16):  # paralel çıkar
                X.append(seri); y.append(etiket)  # biriktir
    return np.stack(X), np.array(y, dtype=np.int64)  # [N, T, D] + etiketler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,  # argüman ayrıştırıcı
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kosu', type=int, default=5)  # koşu sayısı
    parser.add_argument('--turler', nargs='+', default=['pitch', 'noise'],  # artırma türleri
                        choices=['pitch', 'noise'])
    parser.add_argument('--islemler', type=int, default=14)  # paralel süreç
    args = parser.parse_args()  # argümanları oku

    # AĞIR importlar SADECE burada (işçiler bu koda hiç girmez -> torch yüklemez).
    import pandas as pd  # manifest
    import torch  # cihaz + eğitim
    from final.ablasyon import MODEL  # kazanan BiGRU
    from final.dataset import Standardizer  # normalizasyon
    from final.models import SeqRNN  # model
    from final.pipeline import SplitSettings, _feature_folds  # bölme + yükleme
    from final.training import (  # eğitim + değerlendirme
        evaluate_arrays, inverse_frequency_weights, train_with_early_stopping)
    from ser.constants import NUM_CLASSES  # sınıf sayısı
    from ser.data.splits import prepare_splits  # bölme
    from ser.evaluate import compute_metrics  # metrikler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    cfg = IntervalConfig(n_intervals=32, interval_ms=200)   # jitter+kontrast (53)
    manifest = pd.read_csv('data/processed/manifest.csv')  # manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))  # bölme ayarı
    tr, va, te = prepare_splits(manifest, ayar, seed=42)  # konuşmacı-bağımsız böl
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)  # sınıf ağırlıkları

    # Val/test: ORİJİNAL öznitelikler (önbellekten). Asla artırılmaz.
    T = _feature_folds({'val': va, 'test': te}, cfg, extract_interval_series,  # val/test öznitelikleri
                       'data/cache/final', workers=8, cache={})
    vx, vy = T['val']; ex, ey = T['test']  # geçerleme + test

    print(f'Eğitim artırılıyor: {len(tr)} × (1 + {len(args.turler)}) = '  # bilgi
          f'{len(tr) * (1 + len(args.turler))} seri (paralel, {args.islemler} işçi)...',
          flush=True)
    tx, ty = artirilmis_egitim(tr, cfg, args.turler, args.islemler)  # genişletilmiş eğitim
    print(f'Genişletilmiş eğitim: {tx.shape}', flush=True)  # boyutu bildir

    olcek = Standardizer.fit(tx, feature_axis=2)  # (genişletilmiş) eğitimden normalizasyon
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)  # normalize et

    accs, f1s = [], []  # koşu sonuçları
    for k in range(args.kosu):  # her koşu
        torch.manual_seed(k)  # torch tohumu
        torch.cuda.manual_seed_all(k)  # GPU tohumu
        outcome = train_with_early_stopping(  # eğit
            SeqRNN(cfg.feature_dim, NUM_CLASSES, MODEL), tx_s, ty, vx_s, vy,
            MODEL.optim, num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)  # test
        m = compute_metrics(ey, prob.argmax(axis=1))  # metrikler
        accs.append(m['accuracy']); f1s.append(m['macro_f1'])  # biriktir
        print(f'  koşu {k}: test_acc={m["accuracy"]:.4f}', flush=True)  # koşuyu bildir

    import json  # JSON yazma
    ort = {'turler': args.turler, 'kosu': args.kosu,  # özet sözlüğü
           'test_acc_ort': float(np.mean(accs)), 'test_acc_std': float(np.std(accs)),
           'macro_f1_ort': float(np.mean(f1s)), 'artirmasiz_taban': 0.6694}
    Path('final/outputs/cremad/rnn').mkdir(parents=True, exist_ok=True)  # klasörü oluştur
    Path('final/outputs/cremad/rnn/ses_artirma_deney.json').write_text(  # JSON'a yaz
        json.dumps(ort, indent=2), encoding='utf-8')

    print(f'\n===== SES ARTIRMA ({"+".join(args.turler)}) — {args.kosu} koşu =====', flush=True)  # başlık
    print(f'  test acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}')  # ortalama ± std
    print(f'  macro-F1 = {np.mean(f1s):.4f}')  # macro-F1
    print(f'  KIYAS: artırmasız tek model 5-koşu ort. = 0.6694')  # kıyas
    fark = float(np.mean(accs)) - 0.6694  # artırmanın farkı
    sonuc = 'İŞE YARADI' if fark > 0.003 else ('nötr/gürültü' if abs(fark) <= 0.003 else 'ZARAR VERDİ')  # yorum
    print(f'  fark: {fark:+.4f}  ->  {sonuc}')  # sonucu bildir


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
