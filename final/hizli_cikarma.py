# Süreç-paralel öznitelik ön-çıkarma — CPU'yu GERÇEKTEN dolduran sürüm.
#
# Neden var: Python'da iş parçacıkları (thread) GIL kilidi yüzünden aynı anda tek çekirdek çalıştırır; 24 thread ile bile CPU %7'de kalıyordu. SÜREÇLER (process) ise her biri kendi Python'una sahip olduğundan kilidi paylaşmaz: 16 süreç = 16 çekirdek gerçekten çalışır.
#
# Bu araç, rastgele aramanın kullanacağı TÜM aralık düzenlerinin özniteliklerini önbelleğe yazar. Çekirdek koda dokunmaz; yalnızca data/cache/final'i doldurur. Sonrasında otomatik_arama.py çıkarma yapmadan, saf GPU hızında koşar.
#
# Örnek:
# python final/hizli_cikarma.py --islemler 14

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Rastgele aramanın örnekleme havuzuyla AYNI uzay (5 x 5 = 25 düzen).
ARALIK_SAYILARI = (16, 24, 32, 40, 48)
ARALIK_GENISLIKLERI = (150, 200, 250, 300, 400)

_CACHE = None   # her işçi süreçte initializer doldurur


def _isci_kur(cache_root: str) -> None:
    # İşçi süreç başlarken bir kez: BLAS'ı tek iş parçacığına sabitle.
    #
    # Her süreç kendi içinde numpy/BLAS'a çok iş parçacığı açarsa 16 süreç x N thread birbirini boğar; süreç başına 1 thread en hızlısıdır.

    global _CACHE
    for anahtar in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS'):
        os.environ[anahtar] = '1'
    _CACHE = cache_root


def _cikar(gorev) -> int:  # Tek (dosya, düzen) çifti için özniteliği hesaplayıp önbelleğe yazar.

    yol, cfg_sozluk = gorev
    from final.dataset import load_or_extract
    from final.features import IntervalConfig, extract_interval_series

    cfg = IntervalConfig(**cfg_sozluk)
    load_or_extract(yol, _CACHE, cfg.fingerprint,
                    lambda p: extract_interval_series(p, cfg), cfg.shape)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--islemler', type=int, default=14,
                        help='Paralel süreç sayısı (fiziksel çekirdek civarı iyi).')
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--cache-root', default='data/cache/final')
    parser.add_argument('--sadece', nargs=2, type=int, metavar=('N', 'MS'),
                        help='Yalnız tek aralık düzeni çıkar, ör: --sadece 32 200')
    args = parser.parse_args()

    import pandas as pd
    from final.dataset import feature_cache_path
    from final.features import IntervalConfig

    # CREMA-D'nin TÜM dosyaları (hangi bölme olursa olsun önbellek ortak).
    manifest = pd.read_csv(args.manifest)
    yollar = manifest[manifest['corpus'] == 'cremad']['path'].astype(str).tolist()

    # Hangi düzenler çıkarılacak: tümü mü, yoksa yalnız --sadece verilen mi.
    if args.sadece:
        duzenler = [(args.sadece[0], args.sadece[1])]
    else:
        duzenler = list(product(ARALIK_SAYILARI, ARALIK_GENISLIKLERI))

    # Eksik (henüz önbelleğe yazılmamış) işleri listele.
    gorevler = []
    for n, ms in duzenler:
        cfg = IntervalConfig(n_intervals=n, interval_ms=ms)
        sozluk = {'n_intervals': n, 'interval_ms': ms}
        eksik = [y for y in yollar
                 if not feature_cache_path(y, args.cache_root, cfg.fingerprint).is_file()]
        gorevler += [(y, sozluk) for y in eksik]
        print(f'{n:>2} aralık x {ms:>3} ms: {len(yollar) - len(eksik):>5} hazır, '
              f'{len(eksik):>5} eksik')

    if not gorevler:
        print('\nHer şey zaten önbellekte — yapılacak iş yok.')
        return

    print(f'\nToplam {len(gorevler):,} çıkarma işi, {args.islemler} paralel süreç. '
          f'Başlıyor...'.replace(',', '.'))
    basla = time.perf_counter()
    bitti = 0
    with ProcessPoolExecutor(max_workers=args.islemler,
                             initializer=_isci_kur,
                             initargs=(args.cache_root,)) as havuz:
        # chunksize: görevleri paketler halinde dağıt -> süreçler arası mesaj
        # trafiği azalır, verim artar.
        for _ in havuz.map(_cikar, gorevler, chunksize=64):
            bitti += 1
            if bitti % 2000 == 0 or bitti == len(gorevler):
                gecen = time.perf_counter() - basla
                hiz = bitti / max(gecen, 1e-9)
                kalan = (len(gorevler) - bitti) / max(hiz, 1e-9)
                print(f'  {bitti:>6}/{len(gorevler)} | {hiz:,.0f} iş/sn | '
                      f'kalan ~{kalan/60:.1f} dk'.replace(',', '.'))
    print(f'\nBitti: {len(gorevler):,} iş, {(time.perf_counter()-basla)/60:.1f} dakika.'
          .replace(',', '.'))
    print('Artık otomatik_arama.py çıkarma beklemeden, saf GPU hızında koşar.')


if __name__ == '__main__':
    main()
