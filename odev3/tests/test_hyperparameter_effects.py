"""Tamamlanmış doğrulama aramalarının betimsel özetlerine ait testler.

hyperparameter_effects.py'nin dört davranışı sınanır: stability satırlarının
analiz dışında tutulması, grup istatistiklerinin elle doğrulanabilir
doğruluğu, bozuk girdilerin (eksik sütun, kopya config_id) reddedilmesi ve
tam analizin tüm CSV/JSON çıktılarını doğru içerikle yazması.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from odev3 import hyperparameter_effects as effects


def _validation_frame() -> pd.DataFrame:
    # Dört satırlık sahte doğrulama tablosu. Kurgunun püf noktaları:
    # - İlk iki satır aynı learning_rate (0.001) ile bir GRUP oluşturur.
    # - Üçüncü satır farklı lr + aktivasyon + batch_norm ile ikinci grupları açar.
    # - Dördüncü satır 'stability' aşamasıdır ve config_id'si 2. satırla
    #   AYNIDIR (kasıtlı): analiz bu satırı dışlamalı; dahil etseydi 0.9'luk
    #   skor lr=0.001 grubunun ortalamasını yapay olarak şişirirdi.
    common = {
        'batch_size': 64,
        'patience': 8,
        'hidden_dims': '512-256',
        'activation': 'relu',
        'batch_norm': True,
        'dropout': 0.3,
        'weight_decay': 0.0001,
    }
    return pd.DataFrame(
        [
            {
                **common,
                'trial': 1,
                'search_stage': 'screening',
                'config_id': 'a',
                'learning_rate': 0.001,
                'val_macro_f1': 0.4,
            },
            {
                **common,
                'trial': 2,
                'search_stage': 'refinement',
                'config_id': 'b',
                'learning_rate': 0.001,
                'val_macro_f1': 0.5,
            },
            {
                **common,
                'trial': 3,
                'search_stage': 'refinement',
                'config_id': 'c',
                'learning_rate': 0.0001,
                'activation': 'gelu',
                'batch_norm': False,
                'val_macro_f1': 0.3,
            },
            {
                **common,
                'trial': 4,
                'search_stage': 'stability',
                'config_id': 'b',
                'learning_rate': 0.001,
                'val_macro_f1': 0.9,
            },
        ]
    )


def test_validation_search_rows_excludes_repeated_stability_seed() -> None:
    '''Süzme, stability tekrarını atmalı; kalan 3 satırın config_id'leri benzersiz olmalı.'''

    filtered = effects.validation_search_rows(_validation_frame())

    assert len(filtered) == 3
    assert set(filtered['search_stage']) == {'screening', 'refinement'}
    assert filtered['config_id'].is_unique


def test_parameter_effects_report_group_counts_and_descriptive_spread() -> None:
    '''Grup istatistikleri elle hesaplanan değerlere eşit olmalı; genel bakış en iyi/kötüyü seçmeli.

    lr=0.001 grubunun iki koşusu (0.4 ve 0.5): ortalama 0.45, std 0.05
    (ddof=0). Genel bakışta en iyi değer 0.001, en kötü 0.0001 ve fark
    0.45-0.30=0.15 olmalı. batch_norm için hem true hem false denendiği de
    doğrulanır (kanonik metin 'true'/'false').
    '''

    rows = effects.parameter_effect_rows(_validation_frame())
    learning_rate = [row for row in rows if row['parameter'] == 'learning_rate']

    assert len(learning_rate) == 2
    best_group = next(row for row in learning_rate if row['value'] == '0.001')
    assert best_group['runs'] == 2
    assert best_group['val_macro_f1_mean'] == pytest.approx(0.45)
    assert best_group['val_macro_f1_std'] == pytest.approx(0.05)

    overview = effects.parameter_effect_overview(rows)
    learning_rate_overview = next(row for row in overview if row['parameter'] == 'learning_rate')
    assert learning_rate_overview['best_value'] == '0.001'
    assert learning_rate_overview['worst_value'] == '0.0001'
    assert learning_rate_overview['mean_macro_f1_spread'] == pytest.approx(0.15)
    batch_norm = next(row for row in overview if row['parameter'] == 'batch_norm')
    assert set(batch_norm['tested_values'].split(' | ')) == {'false', 'true'}


def test_validation_search_rows_rejects_missing_or_duplicate_configs() -> None:
    '''Eksik sütun ve kopya config_id, anlaşılır mesajlarla ayrı ayrı reddedilmeli.

    Kopya config_id özellikle tehlikelidir: aynı konfigürasyonun iki kez
    sayılması grup ortalamalarını sessizce çarpıtırdı.
    '''

    missing = _validation_frame().drop(columns=['dropout'])
    with pytest.raises(ValueError, match='missing columns'):
        effects.validation_search_rows(missing)

    duplicated = _validation_frame()
    duplicated.loc[2, 'config_id'] = 'a'
    with pytest.raises(ValueError, match='must be unique'):
        effects.validation_search_rows(duplicated)


def test_analysis_writes_per_dataset_csv_and_json_artifacts(
    tmp_path: Path,
) -> None:
    '''Uçtan uca analiz: iki korpus için CSV'ler yazılmalı, summary.json diskteki haliyle eşleşmeli.

    Doğrulananlar: dahil edilen aşama listesi, metodolojik sınır cümlesinin
    varlığı, korpus başına 3 konfigürasyon + 1 dışlanmış stability satırı,
    her parametre için bir genel bakış satırı ve diske yazılan JSON'un
    fonksiyonun döndürdüğü sözlükle birebir aynı olması.
    '''

    output_root = tmp_path / 'outputs'
    for corpus in ('cremad', 'meld'):
        corpus_dir = output_root / corpus
        corpus_dir.mkdir(parents=True)
        _validation_frame().to_csv(corpus_dir / 'validation_results.csv', index=False)
    analysis_root = tmp_path / 'analysis'

    summary = effects.analyze_hyperparameter_effects(
        output_root=output_root,
        analysis_root=analysis_root,
    )

    assert summary['included_search_stages'] == ['screening', 'refinement']
    assert 'not a balanced full-factorial' in summary['important_limitation']
    for corpus in ('cremad', 'meld'):
        result = summary['per_dataset'][corpus]
        assert result['configuration_trials'] == 3
        assert result['excluded_stability_trials'] == 1
        assert len(result['overview']) == len(effects.PARAMETERS)
        assert (analysis_root / corpus / 'parameter_effects.csv').is_file()
        assert (analysis_root / corpus / 'parameter_effect_overview.csv').is_file()
    saved = json.loads((analysis_root / 'summary.json').read_text(encoding='utf-8'))
    assert saved == summary
