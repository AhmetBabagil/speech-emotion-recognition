"""Tamamlanmış doğrulama denemelerinden betimsel hiperparametre özetleri.

Bu modül, hiperparametre araması BİTTİKTEN sonra devreye girer: her deneme
(trial) için kaydedilen doğrulama macro-F1 skorlarını okur ve "learning rate
1e-3 olan denemelerin ortalaması kaçtı?" gibi gruplanmış özet tablolar üretir.

Önemli metodolojik uyarı (summary.json'a da yazılır): arama dengeli bir
tam-faktöriyel deney DEĞİLDİR — parametreler adaylar arasında birlikte
değişir. Bu yüzden buradaki farklar "betimsel ilişkilerdir"; tek başına
nedensel etki olarak yorumlanamazlar. Kod bu sınırı bilerek raporlar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ser.utils import ensure_dir


# Çıktı şemasının sürümü: ileride format değişirse eski özet dosyaları
# ayırt edilebilsin diye yazılır.
PROTOCOL_VERSION = 1
# Analize dahil edilen arama aşamaları. 'stability' aşaması bilerek dışarıda:
# o aşamada AYNI konfigürasyon farklı tohumlarla tekrar eğitilir; dahil
# edilseydi kazanan ayar gruplara fazladan ağırlıkla girer ve ortalamaları
# çarpıtırdı.
SEARCH_STAGES = ('screening', 'refinement')
# Özetlenen hiperparametreler ve rapor tablosunda görünecek etiketleri.
# (Etiket metinleri çalışma zamanı çıktısıdır; CSV'ye aynen yazılır.)
PARAMETERS = (
    ('learning_rate', 'Learning rate'),
    ('hidden_dims', 'Gizli katman yapısı'),
    ('batch_size', 'Batch boyutu'),
    ('patience', 'Patience'),
    ('activation', 'Aktivasyon'),
    ('batch_norm', 'Batch normalization'),
    ('dropout', 'Dropout'),
    ('weight_decay', 'Weight decay'),
)


def _canonical_value(value: Any) -> str:
    # Farklı tipteki hücre değerlerini tek biçimli metne çevirir; böylece
    # gruplama anahtarları ("0.0003" vs "3e-04" gibi) tutarlı görünür.
    # Sıra önemli: numpy bool'ları int'in alt sınıfı gibi davranabildiği için
    # önce bool kontrol edilir.
    if isinstance(value, (bool, np.bool_)):
        return 'true' if bool(value) else 'false'
    if isinstance(value, (float, np.floating)):
        # .10g biçimi gereksiz sıfırları atar ama hassasiyeti korur.
        return f'{float(value):.10g}'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def validation_search_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Tekrarlı kararlılık tohumları olmadan, benzersiz konfigürasyon-arama satırlarını döndürür.

    Üç güvenlik kontrolü yapar:
    1. Gerekli tüm sütunlar var mı? (Eksikse hangileri eksik, açıkça söylenir.)
    2. screening/refinement dışındaki satırlar atıldıktan sonra veri kaldı mı?
    3. Kalan satırlarda config_id benzersiz mi ve skorlar sonlu mu?
    Bu kontroller, bozuk/yarım bir CSV'nin sessizce yanlış özet üretmesini
    engeller.
    """

    required = {
        'trial',
        'search_stage',
        'config_id',
        'val_macro_f1',
        *(name for name, _label in PARAMETERS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'validation results are missing columns: {missing}')
    filtered = frame[frame['search_stage'].isin(SEARCH_STAGES)].copy()
    if filtered.empty:
        raise ValueError('validation results contain no screening/refinement rows.')
    if filtered['config_id'].duplicated().any():
        raise ValueError('screening/refinement config_id values must be unique.')
    if not np.isfinite(filtered['val_macro_f1'].to_numpy(dtype=np.float64)).all():
        raise ValueError('validation macro-F1 values must be finite.')
    return filtered


def parameter_effect_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Doğrulama macro-F1'ini, denenen her hiperparametre değeri bazında toplar.

    Örnek çıktı satırı: "dropout=0.5 olan 4 denemenin ortalaması 0.48,
    std'si 0.02, en iyisi trial 7". Her parametre için, o parametrenin aldığı
    her değer bir grup olur ve grubun istatistikleri hesaplanır.
    """

    search_rows = validation_search_rows(frame)
    rows = []
    # parameter_order: tabloda parametrelerin hep aynı sırada görünmesi için
    # 1'den başlayan sabit bir sıra numarası.
    for parameter_order, (parameter, label) in enumerate(PARAMETERS, start=1):
        # dropna=False: NaN değerli gruplar da (olursa) kaybolmasın;
        # sort=False: grupların CSV'deki doğal karşılaşma sırası korunur.
        for value, group in search_rows.groupby(parameter, dropna=False, sort=False):
            scores = group['val_macro_f1'].to_numpy(dtype=np.float64)
            # idxmax orijinal DataFrame indeksini döndürür; bu indeksle ana
            # tablodan "en iyi denemenin" trial numarasını çekeceğiz.
            best_index = group['val_macro_f1'].idxmax()
            rows.append(
                {
                    'parameter_order': parameter_order,
                    'parameter': parameter,
                    'parameter_label': label,
                    'value': _canonical_value(value),
                    'runs': len(group),
                    'val_macro_f1_mean': float(scores.mean()),
                    # ddof=0: örneklemin kendi std'si (popülasyon formülü);
                    # grup betimsel olarak özetlendiği için uygundur.
                    'val_macro_f1_std': float(scores.std(ddof=0)),
                    'val_macro_f1_median': float(np.median(scores)),
                    'val_macro_f1_min': float(scores.min()),
                    'val_macro_f1_max': float(scores.max()),
                    'best_trial': int(frame.loc[best_index, 'trial']),
                    'search_stages': ','.join(sorted({str(stage) for stage in group['search_stage']})),
                }
            )
    # Sıralama üç anahtarlı: önce parametre sırası, sonra ortalamaya göre
    # azalan (eksi işaretiyle), eşitlikte değere göre alfabetik. Böylece her
    # parametrenin en iyi değeri kendi bloğunun tepesinde görünür.
    rows.sort(
        key=lambda row: (
            int(row['parameter_order']),
            -float(row['val_macro_f1_mean']),
            str(row['value']),
        )
    )
    return rows


def parameter_effect_overview(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Her parametre için en yüksek ve en düşük betimsel grup ortalamasını seçer.

    Detay tablosunu (parameter_effect_rows çıktısı) parametre başına tek
    satıra indirger: "bu parametrenin en iyi değeri X (ortalama F1=...),
    en kötüsü Y, aradaki fark (spread) şu kadar". Spread büyükse o
    parametrenin skoru ne kadar oynatabildiği hakkında fikir verir — ama yine
    betimseldir, nedensellik iddia etmez.
    """

    overviews = []
    for parameter_order, (parameter, label) in enumerate(PARAMETERS, start=1):
        parameter_rows = [row for row in rows if row['parameter'] == parameter]
        if not parameter_rows:
            raise ValueError(f'no grouped rows found for {parameter}.')
        # En iyi grubu seçerken beraberlik bozucular: önce ortalama, sonra
        # grubun max skoru, en son değerin alfabetik hali (determinizm için).
        ordered = sorted(
            parameter_rows,
            key=lambda row: (
                -float(row['val_macro_f1_mean']),
                -float(row['val_macro_f1_max']),
                str(row['value']),
            ),
        )
        best = ordered[0]
        worst = ordered[-1]
        overviews.append(
            {
                'parameter_order': parameter_order,
                'parameter': parameter,
                'parameter_label': label,
                'tested_value_count': len(parameter_rows),
                'tested_values': ' | '.join(sorted(str(row['value']) for row in parameter_rows)),
                'best_value': best['value'],
                'best_runs': int(best['runs']),
                'best_mean_macro_f1': float(best['val_macro_f1_mean']),
                'best_std_macro_f1': float(best['val_macro_f1_std']),
                'worst_value': worst['value'],
                'worst_runs': int(worst['runs']),
                'worst_mean_macro_f1': float(worst['val_macro_f1_mean']),
                'mean_macro_f1_spread': float(best['val_macro_f1_mean'] - worst['val_macro_f1_mean']),
                # Grup boyutları: bir grupta 1, diğerinde 15 deneme varsa
                # karşılaştırma zayıftır; okuyucu bunu görebilsin diye yazılır.
                'minimum_group_runs': min(int(row['runs']) for row in parameter_rows),
                'maximum_group_runs': max(int(row['runs']) for row in parameter_rows),
            }
        )
    return overviews


def analyze_hyperparameter_effects(
    *,
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    output_root: str | Path = 'odev3/outputs',
    analysis_root: str | Path = 'odev3/hyperparameter_effects',
) -> dict[str, Any]:
    # Ana giriş noktası: her korpusun validation_results.csv dosyasını okur,
    # iki özet tabloyu (detay + genel bakış) CSV olarak yazar ve hepsini tek
    # bir summary.json'da toplar.
    output_root = Path(output_root)
    analysis_root = ensure_dir(analysis_root)
    per_dataset = {}
    for corpus in corpora:
        source_path = output_root / corpus / 'validation_results.csv'
        if not source_path.is_file():
            raise FileNotFoundError(f'Missing validation results: {source_path}')
        frame = pd.read_csv(source_path)
        search_rows = validation_search_rows(frame)
        grouped_rows = parameter_effect_rows(frame)
        overview_rows = parameter_effect_overview(grouped_rows)
        corpus_dir = ensure_dir(Path(analysis_root) / corpus)
        pd.DataFrame(grouped_rows).to_csv(
            corpus_dir / 'parameter_effects.csv',
            index=False,
        )
        pd.DataFrame(overview_rows).to_csv(
            corpus_dir / 'parameter_effect_overview.csv',
            index=False,
        )
        per_dataset[corpus] = {
            'source': str(source_path),
            'configuration_trials': len(search_rows),
            'excluded_stability_trials': int((frame['search_stage'] == 'stability').sum()),
            'grouped_rows': grouped_rows,
            'overview': overview_rows,
        }

    # Özet JSON: yöntemin ne olduğu ve — çok önemli — ne OLMADIĞI
    # (important_limitation alanı) makine tarafından okunabilir şekilde kalıcılaşır.
    summary = {
        'protocol_version': PROTOCOL_VERSION,
        'analysis_type': 'descriptive grouped validation macro-F1 summary',
        'included_search_stages': list(SEARCH_STAGES),
        'excluded_search_stage': 'stability',
        'important_limitation': (
            'The search is not a balanced full-factorial experiment. Parameters '
            'co-vary across configurations, so grouped differences are descriptive '
            'associations and must not be interpreted as isolated causal effects.'
        ),
        'per_dataset': per_dataset,
    }
    # ensure_ascii=False: Türkçe etiketler JSON'da okunabilir kalsın
    # (kaçış dizisi ı yerine gerçek karakter yazılsın).
    (Path(analysis_root) / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return summary


def main() -> None:
    # Bağımsız CLI: analiz, deney bittikten sonra tek başına da çalıştırılabilir.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpora', nargs='+', default=['cremad', 'meld'])
    parser.add_argument('--output-root', default='odev3/outputs')
    parser.add_argument('--analysis-root', default='odev3/hyperparameter_effects')
    args = parser.parse_args()
    analyze_hyperparameter_effects(
        corpora=tuple(args.corpora),
        output_root=args.output_root,
        analysis_root=args.analysis_root,
    )


if __name__ == '__main__':
    main()
