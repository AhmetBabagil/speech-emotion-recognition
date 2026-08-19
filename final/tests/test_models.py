# Final modelleri ve artırmaları için ileri-geçiş (forward) boyut testleri.
#
# Amaç: her model rastgele bir yığını alıp doğru boyutta logit üretebiliyor mu? Bu basit kontroller, mimari değişikliklerinde boyut uyuşmazlıklarını anında yakalar.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import pytest  # test çatısı
import torch  # tensörler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from final.augment import FeatureNoise, SpecMask  # noqa: E402  # veri artırmaları
from final.models import (  # noqa: E402  # modeller + ayarlar
    CNNConfig,
    MelCNN,
    OptimSettings,
    RNNConfig,
    SeqRNN,
)


def test_cnn_forward_shape() -> None:  # CNN: [B, mels, T] girdiden [B, 6] logit üretmeli.

    model = MelCNN(6, CNNConfig(channels=(8, 16)))  # küçük CNN kur
    logits = model(torch.randn(4, 64, 128))  # rastgele yığın ver
    assert logits.shape == (4, 6)  # çıktı 4 örnek x 6 sınıf olmalı


@pytest.mark.parametrize('pooling', ['last', 'mean', 'max', 'attn'])  # dört havuzlama türünü ayrı ayrı dene
def test_rnn_forward_shape_all_poolings(pooling: str) -> None:  # RNN: dört havuzlama türünün DÖRDÜ de doğru boyutta çıktı vermeli.

    config = RNNConfig(hidden_size=32, num_layers=1, pooling=pooling,  # küçük RNN ayarı
                       optim=OptimSettings())
    model = SeqRNN(44, 6, config)  # 44 boyutlu girdi için RNN
    logits = model(torch.randn(4, 24, 44))  # rastgele seri ver
    assert logits.shape == (4, 6)  # çıktı 4 x 6 olmalı


def test_rnn_rejects_bad_pooling() -> None:  # Geçersiz havuzlama adı sessizce kabul edilmemeli, hata vermeli.

    with pytest.raises(ValueError):  # ValueError beklenir
        RNNConfig(pooling='sum').validate()  # 'sum' geçersiz -> hata


def test_spec_mask_zeroes_but_keeps_shape() -> None:  # SpecAugment: boyutu korumalı, girdiyi bozmamalı, gerçekten maskelemeli.

    batch = torch.ones(3, 64, 128)  # tümü 1 olan yığın
    masked = SpecMask()(batch)  # maskele
    assert masked.shape == batch.shape  # boyut korunmalı
    assert (batch == 1.0).all()      # orijinal tensör yerinde değişmemeli (clone)
    assert (masked == 0.0).sum() > 0  # en az bir bölge sıfırlanmış olmalı


def test_feature_noise_shape() -> None:  # Gürültü artırması: boyut aynı kalmalı, çıktı gerçekten gürültülü olmalı.

    torch.manual_seed(0)  # tekrarlanabilirlik
    batch = torch.zeros(3, 24, 44)  # tümü 0 olan yığın
    noisy = FeatureNoise(std=0.1)(batch)  # gürültü ekle
    assert noisy.shape == batch.shape  # boyut korunmalı
    assert noisy.std() > 0.05   # sıfır tensöre gürültü eklendiyse std > 0
