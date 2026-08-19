# Demo: kaydedilmiş final modellerini WAV dosyalarında çalıştırır, iki yöntemin çıktısını yan yana gösterir.
#
# Yönergenin 8. bölümünün birebir karşılığı: girdiler ya iyi/kötü örnek dizininden RASTGELE seçilir ya da komut satırında ADIYLA/YOLUYLA verilir; her girdi iki yöntemden geçirilip sonuçlarıyla gösterilir.
#
# Örnekler:
# python final/demo.py --rastgele 3
# python final/demo.py final/demo_ornekleri/1091_IEO_FEA_HI.wav
# python final/demo.py --dizin final/demo_ornekleri --rastgele 5

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
from pathlib import Path  # dosya yolları
import random  # rastgele örnek seçimi
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # olasılıkları sıralamak için
import torch  # noqa: E402  # cihaz

from final.predict import load_checkpoint, predict  # noqa: E402  # model yükleme + tahmin
from ser.constants import CANONICAL_EMOTIONS, CREMAD_CODE_TO_CANONICAL  # noqa: E402  # duygu adları/kodları

# Test kümesinden seçilmiş 6 iyi + 4 kötü örneğin durduğu varsayılan dizin.
DEFAULT_DIR = Path('final/demo_ornekleri')  # varsayılan örnek dizini

# Her yöntem için tercih sırası: geliştirme aşamasının modeli varsa onu,
# yoksa arama kazananını kullan.
MODEL_PATHS = {  # yöntem adı -> (tercih edilen model, yedek model)
    'Yöntem 1 (CNN)': (
        'final/outputs/cremad/cnn/improved_model.pt',  # önce geliştirilmiş
        'final/outputs/cremad/cnn/winner_model.pt',  # yoksa arama kazananı
    ),
    'Yöntem 2 (BiGRU)': (
        'final/outputs/cremad/rnn/improved_model.pt',  # önce geliştirilmiş
        'final/outputs/cremad/rnn/winner_model.pt',  # yoksa arama kazananı
    ),
}


def true_label_from_name(path: Path) -> str | None:
    # CREMA-D dosya adının 3. parçası duygu kodudur (ör. ..._ANG_... -> angry).
    #
    # Bu sayede demo, tahminin doğru mu yanlış mı olduğunu (✓/✗) etiketli dosyalarda otomatik gösterebilir.

    parts = path.stem.split('_')  # dosya adını '_' ile böl
    if len(parts) >= 3:  # en az 3 parça varsa
        return CREMAD_CODE_TO_CANONICAL.get(parts[2].upper())  # 3. parça = duygu kodu -> ad
    return None  # ad çözülemezse bilinmiyor


def load_methods(device: torch.device) -> dict:  # İki yöntemin modellerini diskten yükler.

    methods = {}  # yöntem adı -> yüklenmiş model paketi
    for name, candidates in MODEL_PATHS.items():  # her yöntem için
        # Tercih listesindeki ilk mevcut dosyayı seç.
        chosen = next((p for p in candidates if Path(p).is_file()), None)  # var olan ilk model
        if chosen is None:  # hiçbiri yoksa
            raise FileNotFoundError(
                f'{name} için model bulunamadı; önce final/run_experiment.py çalıştırın.'  # hata
            )
        methods[name] = load_checkpoint(chosen, device)  # modeli yükle
    return methods  # yüklenmiş yöntemler


def run_demo(paths: list[Path], methods: dict, device: torch.device) -> None:  # Her dosyayı iki yöntemden geçirir; tahmin + ✓/✗ + ilk 3 olasılığı basar.

    for path in paths:  # her ses dosyası için
        truth = true_label_from_name(path)  # gerçek duygu (dosya adından)
        print(f'\n{path.name}   (gerçek: {truth or "bilinmiyor"})')  # dosya + gerçek etiket
        for name, (model, feature_cfg, extract_fn, standardizer) in methods.items():  # her yöntem için
            row = predict([path], model, feature_cfg, extract_fn, standardizer, device)[0]  # olasılıklar
            order = np.argsort(row)[::-1]  # olasılıkları büyükten küçüğe sırala
            guess = CANONICAL_EMOTIONS[order[0]]  # en yüksek olasılık = tahmin
            isaret = '' if truth is None else ('  ✓' if guess == truth else '  ✗')  # doğru mu işareti
            top3 = ', '.join(f'{CANONICAL_EMOTIONS[i]}={row[i]:.2f}' for i in order[:3])  # ilk 3 olasılık
            print(f'  {name}: {guess}{isaret}   ({top3})')  # yöntemin sonucunu yazdır


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('girdi', nargs='*', help='WAV dosya yolları (adıyla çağırma).')  # elle verilen dosyalar
    parser.add_argument('--dizin', default=str(DEFAULT_DIR),  # örnek dizini
                        help='İyi/kötü örneklerin bulunduğu dizin.')
    parser.add_argument('--rastgele', type=int, default=0,  # rastgele kaç örnek
                        help='Dizinden rastgele bu kadar örnek seç.')
    parser.add_argument('--seed', type=int, help='Rastgele seçim için tohum.')  # seçim tohumu
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')  # cihaz
    args = parser.parse_args()  # argümanları oku

    # Girdi listesi: elle verilenler + (istenirse) dizinden rastgele seçilenler.
    paths = [Path(p) for p in args.girdi]  # elle verilen yollar
    if args.rastgele > 0:  # rastgele seçim istendiyse
        havuz = sorted(Path(args.dizin).glob('*.wav'))  # dizindeki tüm .wav'lar
        if not havuz:  # hiç .wav yoksa
            parser.error(f'{args.dizin} içinde .wav bulunamadı.')  # hata
        rng = random.Random(args.seed)   # seed verilirse seçim tekrarlanabilir
        paths.extend(rng.sample(havuz, min(args.rastgele, len(havuz))))  # rastgele seç ve ekle
    if not paths:  # hiç dosya yoksa
        parser.error('En az bir dosya verin ya da --rastgele N kullanın.')  # hata
    eksik = [p for p in paths if not p.is_file()]  # var olmayan dosyalar
    if eksik:  # eksik varsa
        parser.error(f'Dosya bulunamadı: {eksik}')  # hata

    if args.device == 'auto':  # otomatik cihaz
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    else:  # elle verildiyse
        device = torch.device(args.device)  # o cihaz
    methods = load_methods(device)  # modelleri yükle
    run_demo(paths, methods, device)  # demoyu koştur


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
