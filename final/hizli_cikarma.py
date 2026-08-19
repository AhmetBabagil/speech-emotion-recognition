# Süreç-paralel öznitelik ön-çıkarma — CPU'yu GERÇEKTEN dolduran sürüm.
#
# Neden var: Python'da iş parçacıkları (thread) GIL kilidi yüzünden aynı anda tek çekirdek çalıştırır; 24 thread ile bile CPU %7'de kalıyordu. SÜREÇLER (process) ise her biri kendi Python'una sahip olduğundan kilidi paylaşmaz: 16 süreç = 16 çekirdek gerçekten çalışır.
#
# Bu araç, rastgele aramanın kullanacağı TÜM aralık düzenlerinin özniteliklerini önbelleğe yazar. Çekirdek koda dokunmaz; yalnızca data/cache/final'i doldurur. Sonrasında otomatik_arama.py çıkarma yapmadan, saf GPU hızında koşar.
#
# Örnek: python final/hizli_cikarma.py --islemler 14

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
import os  # ortam değişkenleri (BLAS sabitleme)
import time  # süre ölçümü
from concurrent.futures import ProcessPoolExecutor  # süreç havuzu (GIL'i aşmak için)
from itertools import product  # kartezyen çarpım (tüm düzenler)
from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

# Rastgele aramanın örnekleme havuzuyla AYNI uzay (5 x 5 = 25 düzen).
ARALIK_SAYILARI = (16, 24, 32, 40, 48)  # denenecek aralık sayıları
ARALIK_GENISLIKLERI = (150, 200, 250, 300, 400)  # denenecek aralık genişlikleri (ms)

_CACHE = None   # her işçi süreçte initializer doldurur


def _isci_kur(cache_root: str) -> None:
    # İşçi süreç başlarken bir kez: BLAS'ı tek iş parçacığına sabitle.
    #
    # Her süreç kendi içinde numpy/BLAS'a çok iş parçacığı açarsa 16 süreç x N thread birbirini boğar; süreç başına 1 thread en hızlısıdır.

    global _CACHE  # süreç-genel önbellek yolu
    for anahtar in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',  # BLAS iş parçacıklarını
                    'NUMEXPR_NUM_THREADS'):
        os.environ[anahtar] = '1'  # her birini 1'e sabitle
    _CACHE = cache_root  # önbellek kökünü sakla


def _cikar(gorev) -> int:  # Tek (dosya, düzen) çifti için özniteliği hesaplayıp önbelleğe yazar.

    yol, cfg_sozluk = gorev  # dosya yolu + aralık ayarı
    from final.dataset import load_or_extract  # (süreç içinde import — torch yükleme)
    from final.features import IntervalConfig, extract_interval_series

    cfg = IntervalConfig(**cfg_sozluk)  # ayar nesnesi
    load_or_extract(yol, _CACHE, cfg.fingerprint,  # önbellekte yoksa çıkar ve yaz
                    lambda p: extract_interval_series(p, cfg), cfg.shape)
    return 1  # yalnız ilerleme sayacı için


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--islemler', type=int, default=14,  # paralel süreç sayısı
                        help='Paralel süreç sayısı (fiziksel çekirdek civarı iyi).')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')  # manifest
    parser.add_argument('--cache-root', default='data/cache/final')  # önbellek kökü
    parser.add_argument('--sadece', nargs=2, type=int, metavar=('N', 'MS'),  # tek düzen çıkar
                        help='Yalnız tek aralık düzeni çıkar, ör: --sadece 32 200')
    args = parser.parse_args()  # argümanları oku

    import pandas as pd  # manifest okuma
    from final.dataset import feature_cache_path  # önbellek yolu
    from final.features import IntervalConfig  # ayar

    # CREMA-D'nin TÜM dosyaları (hangi bölme olursa olsun önbellek ortak).
    manifest = pd.read_csv(args.manifest)  # manifesti oku
    yollar = manifest[manifest['corpus'] == 'cremad']['path'].astype(str).tolist()  # cremad dosyaları

    # Hangi düzenler çıkarılacak: tümü mü, yoksa yalnız --sadece verilen mi.
    if args.sadece:  # tek düzen istendiyse
        duzenler = [(args.sadece[0], args.sadece[1])]  # yalnız o
    else:  # aksi
        duzenler = list(product(ARALIK_SAYILARI, ARALIK_GENISLIKLERI))  # tüm 25 düzen

    # Eksik (henüz önbelleğe yazılmamış) işleri listele.
    gorevler = []  # çıkarılacak (dosya, düzen) işleri
    for n, ms in duzenler:  # her düzen için
        cfg = IntervalConfig(n_intervals=n, interval_ms=ms)  # ayar
        sozluk = {'n_intervals': n, 'interval_ms': ms}  # işçiye geçecek sözlük
        eksik = [y for y in yollar  # önbellekte olmayan dosyalar
                 if not feature_cache_path(y, args.cache_root, cfg.fingerprint).is_file()]
        gorevler += [(y, sozluk) for y in eksik]  # eksikleri işlere ekle
        print(f'{n:>2} aralık x {ms:>3} ms: {len(yollar) - len(eksik):>5} hazır, '  # durum
              f'{len(eksik):>5} eksik')

    if not gorevler:  # her şey hazırsa
        print('\nHer şey zaten önbellekte — yapılacak iş yok.')  # bildir
        return  # çık

    print(f'\nToplam {len(gorevler):,} çıkarma işi, {args.islemler} paralel süreç. '  # başlangıç
          f'Başlıyor...'.replace(',', '.'))
    basla = time.perf_counter()  # süre başlat
    bitti = 0  # tamamlanan iş sayısı
    with ProcessPoolExecutor(max_workers=args.islemler,  # süreç havuzu
                             initializer=_isci_kur,  # her işçide BLAS sabitle
                             initargs=(args.cache_root,)) as havuz:
        # chunksize: görevleri paketler halinde dağıt -> süreçler arası mesaj
        # trafiği azalır, verim artar.
        for _ in havuz.map(_cikar, gorevler, chunksize=64):  # paralel çıkar
            bitti += 1  # sayacı artır
            if bitti % 2000 == 0 or bitti == len(gorevler):  # her 2000'de bir bildir
                gecen = time.perf_counter() - basla  # geçen süre
                hiz = bitti / max(gecen, 1e-9)  # iş/saniye
                kalan = (len(gorevler) - bitti) / max(hiz, 1e-9)  # kalan süre tahmini
                print(f'  {bitti:>6}/{len(gorevler)} | {hiz:,.0f} iş/sn | '  # ilerleme
                      f'kalan ~{kalan/60:.1f} dk'.replace(',', '.'))
    print(f'\nBitti: {len(gorevler):,} iş, {(time.perf_counter()-basla)/60:.1f} dakika.'  # bitiş
          .replace(',', '.'))
    print('Artık otomatik_arama.py çıkarma beklemeden, saf GPU hızında koşar.')  # not


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
