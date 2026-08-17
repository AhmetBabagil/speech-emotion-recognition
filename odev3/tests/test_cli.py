'''Ödev 3 komut satırı arayüzünün testleri.

run_experiment.py'nin saf yardımcı fonksiyonları sınanır: sonuç satırı
biçimi, öznitelik konfigürasyonu kurulumu (ödevin 4000 boyut şartı dahil)
ve korpus başına özel ayarların (override) doğru uygulanması. CLI'nin
kendisini (argparse akışını) çalıştırmaya gerek yoktur; mantık bu
fonksiyonlarda toplandığı için onları test etmek yeterlidir.
'''

import pytest

from odev3.run_experiment import (
    _result_line,
    build_feature_config,
    build_feature_configs,
)


def test_result_line_formats_test_metrics() -> None:
    '''Konsol özeti sabit biçimde olmalı: 4 ondalık basamak, "corpus: ..." kalıbı.'''

    result = {'test': {'accuracy': 0.61234, 'macro_f1': 0.54321}}

    line = _result_line('cremad', result)

    assert line == 'cremad: test accuracy=0.6123, macro-F1=0.5432'


def test_build_feature_config_supports_full_utterance_resizing() -> None:
    '''96 kare + resize kombinasyonu geçerli olmalı ve 64*96=6144 boyut üretmeli.'''

    config = build_feature_config(96, 'resize')

    assert config.n_frames == 96
    assert config.frame_strategy == 'resize'
    assert config.vector_size == 6144


def test_build_feature_config_enforces_assignment_minimum() -> None:
    '''Ödevin 4000 boyut alt sınırını ihlal eden ayar (64*62=3968) reddedilmeli.'''

    with pytest.raises(ValueError, match='at least 4000'):
        build_feature_config(62, 'crop_pad')


def test_build_feature_configs_applies_dataset_specific_overrides() -> None:
    '''Korpus başına özel ayarlar varsayılanları alan alan ezmeli (None = varsayılanı kullan).

    Kurgu: cremad yalnızca stratejiyi (resize), meld yalnızca kare sayısını
    (96) değiştirir; değiştirmedikleri alanlar varsayılanda kalmalı.
    '''

    configs = build_feature_configs(
        ('cremad', 'meld'),
        default_frames=64,
        default_strategy='crop_pad',
        frame_overrides={'cremad': None, 'meld': 96},
        strategy_overrides={'cremad': 'resize', 'meld': None},
    )

    assert configs['cremad'].frame_strategy == 'resize'
    assert configs['cremad'].n_frames == 64
    assert configs['meld'].frame_strategy == 'crop_pad'
    assert configs['meld'].n_frames == 96
