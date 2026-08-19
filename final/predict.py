# Eğitilmiş bir final modeliyle bir veya daha çok WAV dosyasının duygusunu tahmin eder.
#
# run_experiment.py / improve.py'nin kaydettiği kontrol noktası (checkpoint) dosyaları; model ağırlıklarını, öznitelik ayarlarını VE normalizasyon parametrelerini birlikte taşır. Bu sayede tahmin sırasında ön işleme, eğitimdekiyle birebir aynı yapılır — "eğitimde başka, tahminde başka ön işleme" hatası yapısal olarak imkânsızdır.
#
# Örnekler:
# python final/predict.py data/raw/cremad/AudioWAV/1001_DFA_ANG_XX.wav
# python final/predict.py kayit.wav --model final/outputs/cremad/rnn/winner_model.pt

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
from pathlib import Path  # dosya yolları
import sys  # import yolu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import numpy as np  # noqa: E402  # diziler + sıralama
import torch  # noqa: E402  # model yükleme + çıkarım

from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import (  # noqa: E402  # öznitelik ayarları + çıkarıcılar
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import CNNConfig, MelCNN, OptimSettings, RNNConfig, SeqRNN  # noqa: E402  # modeller + ayarlar
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # noqa: E402  # duygu adları + sınıf sayısı


def load_checkpoint(path: str | Path, device: torch.device):
    # Diskteki kayıttan (model, öznitelik ayarı, çıkarım fonksiyonu, normalizasyon) dörtlüsünü geri kurar.

    # Kendi eğitim betiklerimizin yazdığı güvenilir yerel dosya; içinde
    # numpy dizileri olduğundan yalnızca-ağırlık (weights_only) yükleyici
    # kullanılamaz.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # kayıt dosyasını yükle
    feature_values = checkpoint['feature_config']  # öznitelik ayarı sözlüğü
    model_values = dict(checkpoint['model_config'])  # model ayarı sözlüğü
    optim = OptimSettings(**model_values.pop('optim'))  # optimizasyon ayarını geri kur

    # Hangi yöntemin kaydı olduğunu ayar anahtarlarından anla:
    # 'n_mels' varsa mel görüntüsü (CNN), yoksa aralık serisi (RNN).
    if 'n_mels' in feature_values:  # CNN kaydı ise
        feature_cfg = MelImageConfig(**feature_values)  # mel ayarı
        extract_fn = extract_mel_image  # mel çıkarıcı
        model = MelCNN(  # CNN modelini kur
            NUM_CLASSES,
            CNNConfig(channels=tuple(model_values['channels']),
                      dropout=model_values['dropout'], optim=optim),
        )
    else:  # RNN kaydı ise
        feature_cfg = IntervalConfig(**feature_values)  # aralık ayarı
        extract_fn = extract_interval_series  # aralık çıkarıcı
        model = SeqRNN(  # RNN modelini kur
            feature_cfg.feature_dim,
            NUM_CLASSES,
            RNNConfig(**model_values, optim=optim),
        )
    model.load_state_dict(checkpoint['state_dict'])  # ağırlıkları yükle
    model.to(device).eval()   # tahmin kipine al (dropout/BN kapanır)
    # Eğitim katmanından öğrenilmiş normalizasyon parametreleri.
    standardizer = Standardizer(  # aynı normalizasyonu geri kur
        mean=np.asarray(checkpoint['standardizer_mean']),  # ortalama
        scale=np.asarray(checkpoint['standardizer_scale']),  # std
        feature_axis=int(checkpoint['feature_axis']),  # eksen
    )
    return model, feature_cfg, extract_fn, standardizer  # dörtlüyü döndür


def predict(paths, model, feature_cfg, extract_fn, standardizer, device):  # Dosya listesini modele verir; [N, 6] boyutlu olasılık matrisi döndürür.

    # Ön işleme zinciri eğitimdekiyle aynı: öznitelik -> normalizasyon -> model.
    features = np.stack([extract_fn(path, feature_cfg) for path in paths])  # her dosyadan öznitelik çıkar
    features = standardizer.transform(features)  # eğitimdeki normalizasyonu uygula
    with torch.no_grad():  # tahminde gradyan yok
        logits = model(torch.from_numpy(features).to(device))  # modelden ham puanlar
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()  # olasılığa çevir
    return probabilities  # [N, 6] olasılık matrisi


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('audio', nargs='+', help='Bir veya daha fazla WAV dosyası.')  # ses dosyaları
    parser.add_argument(  # kullanılacak model
        '--model', default='final/outputs/cremad/cnn/winner_model.pt'
    )
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')  # cihaz
    args = parser.parse_args()  # argümanları oku

    if args.device == 'auto':  # otomatik cihaz
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    else:  # elle verildiyse
        device = torch.device(args.device)  # o cihaz

    model, feature_cfg, extract_fn, standardizer = load_checkpoint(args.model, device)  # modeli yükle
    probabilities = predict(  # dosyaları tahmin et
        args.audio, model, feature_cfg, extract_fn, standardizer, device
    )
    # Her dosya için en olası sınıfı ve ilk 3 olasılığı yazdır.
    for path, row in zip(args.audio, probabilities):  # her dosya için
        order = np.argsort(row)[::-1]  # olasılıkları büyükten küçüğe sırala
        ranking = ', '.join(  # ilk 3 olasılık metni
            f'{CANONICAL_EMOTIONS[i]}={row[i]:.3f}' for i in order[:3]
        )
        print(f'{path}: {CANONICAL_EMOTIONS[order[0]]}  ({ranking})')  # tahmini yazdır


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
