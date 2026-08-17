'''Seçilen Mel temsillerinin, yalnızca doğrulama üzerinde çok tohumlu (multi-seed) teyidi.

Neden bu deney var? Ablasyon çalışması (feature_ablation.py) her adayı TEK
tohumla eğitti. Tek tohumluk bir fark, gerçek bir üstünlük değil "şans"
olabilir: ağırlık başlangıcı ve batch sırası değişince skor da oynar. Bu
modül, her korpusun en iyi iki adayını (ana temsil + en güçlü rakibi) ÜÇ
farklı tohumla yeniden eğitir ve şu soruya cevap arar: "Ablasyonun kazananı,
rastgelelik hesaba katıldığında da hâlâ kazanıyor mu?"

Protokol yine sıkıdır: test sesleri hiç yüklenmez (validation-only), model
konfigürasyonu sabittir, ve tohumlar iki aday için birebir aynıdır — böylece
karşılaştırma "eşleştirilmiş" (paired) olur ve tohum şansı iki tarafı da
eşit etkiler.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from odev3.dataset import FeatureStandardizer, load_feature_matrix
from odev3.feature_ablation import REFERENCE_MODELS
from odev3.features_melspec import MelSpecConfig
from odev3.model import MLPConfig
from odev3.pipeline import _splits_for
from odev3.training import train_with_early_stopping
from ser.constants import NUM_CLASSES
from ser.utils import ensure_dir, get_device


# Çıktı satırlarına yazılan protokol sürümü: format değişirse eski partial
# CSV'ler otomatik olarak "eski" sayılır ve yeniden üretilir.
PROTOCOL_VERSION = 1
# Teyit tohumları: üç sabit, birbirinden uzak değer. Üç tekrar, ortalama ve
# standart sapma hesaplamak için (maliyeti patlatmadan) makul bir alt sınırdır.
CONFIRMATION_SEEDS = (42, 143, 244)
# Grafik başlıklarında kullanılan gösterim adları.
CORPUS_DISPLAY_NAMES = {'cremad': 'CREMA-D', 'meld': 'MELD'}


def confirmation_candidates(
    corpus: str,
) -> tuple[tuple[str, MelSpecConfig], ...]:
    '''Ana temsili ve onun en güçlü tek tohumlu rakibini döndürür.

    Adlandırma bilinçli: 'main_*' ablasyonun tek tohumlu kazananı,
    'challenger_*' ise ikinci gelen aday. paired_candidate_comparison bu
    öneklere ('main_' / 'challenger_') dayanarak iki tarafı ayırt eder.
    Adaylar korpusa göre farklıdır çünkü ablasyon her korpusta farklı bir
    kazanan çıkarmıştır.
    '''

    candidates = {
        'cremad': (
            ('main_64x64_resize', MelSpecConfig(frame_strategy='resize')),
            (
                'challenger_96x64_crop_pad',
                MelSpecConfig(n_mels=96, frame_strategy='crop_pad'),
            ),
        ),
        'meld': (
            ('main_64x64_crop_pad', MelSpecConfig(frame_strategy='crop_pad')),
            (
                'challenger_80x64_resize',
                MelSpecConfig(n_mels=80, frame_strategy='resize'),
            ),
        ),
    }
    if corpus not in candidates:
        raise ValueError(f'Unsupported corpus: {corpus!r}.')
    return candidates[corpus]


def _normalized_model_config(payload: Any) -> str | None:
    # Model konfigürasyonunu kanonik JSON metnine indirger (bkz.
    # feature_ablation._normalized_model_config): CSV'den okunan string ile
    # bellekteki sözlük aynı biçimde karşılaştırılabilsin diye.
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_keys(
    rows: list[dict[str, Any]],
    max_epochs: int,
    model_config: MLPConfig,
) -> set[tuple[str, int]]:
    # Devam etme (resume) anahtarları: bu deneyde bir "iş birimi"
    # (öznitelik parmak izi, tohum) çiftidir — aynı aday üç tohumla üç ayrı
    # iş sayılır. Yalnızca protokol sürümü, epoch limiti ve model ayarı
    # birebir uyuşan satırlar tamamlanmış kabul edilir.
    expected_model = _normalized_model_config(model_config.to_dict())
    return {
        (str(row['feature_fingerprint']), int(row['seed']))
        for row in rows
        if int(row.get('protocol_version', 0)) == PROTOCOL_VERSION
        and int(row['max_epochs']) == max_epochs
        and _normalized_model_config(row.get('model_config')) == expected_model
    }


def _load_partial(path: Path) -> list[dict[str, Any]]:
    # Yarıda kalmış çalıştırmanın ara CSV'sini oku; yoksa boş listeyle başla.
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient='records')


def _matching_rows(
    rows: list[dict[str, Any]],
    *,
    max_epochs: int,
    model_config: MLPConfig,
    candidates: tuple[tuple[str, MelSpecConfig], ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    # Ara CSV'den yalnızca ŞU ANKİ deney tanımıyla uyumlu satırları süzer:
    # doğru protokol sürümü, epoch limiti, model, aday (isim+parmak izi) ve
    # tohum kümesi. Eski/uyumsuz satırlar sessizce elenir; böylece farklı
    # ayarlarla üretilmiş sonuçlar analizlere asla karışmaz.
    expected_model = _normalized_model_config(model_config.to_dict())
    candidate_keys = {
        (name, config.fingerprint) for name, config in candidates
    }
    expected_seeds = set(seeds)
    return [
        row
        for row in rows
        if int(row.get('protocol_version', 0)) == PROTOCOL_VERSION
        and int(row['max_epochs']) == max_epochs
        and _normalized_model_config(row.get('model_config')) == expected_model
        and (str(row['candidate']), str(row['feature_fingerprint']))
        in candidate_keys
        and int(row['seed']) in expected_seeds
    ]


def aggregate_stability_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    '''Öznitelik adaylarını, tohumlar arası ortalama doğrulama macro-F1'e göre sıralar.

    Her (korpus, aday) grubunun üç tohumdaki skorlarını ortalama, std,
    min ve max ile özetler. std burada kilit metrik: küçük std "temsil
    rastgeleliğe dayanıklı", büyük std "sonuç tohuma bağlı, güvenme" demektir.
    '''

    # Satırları aday kimliğine göre grupla. Anahtar, adayı benzersiz
    # tanımlayan tüm alanları içerir; böylece farklı adaylar asla karışmaz.
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row['corpus'],
            row['candidate'],
            row['feature_fingerprint'],
            row['frame_strategy'],
            int(row['n_mels']),
            int(row['n_frames']),
            int(row['vector_size']),
        )
        groups.setdefault(key, []).append(row)

    aggregates = []
    for key, candidate_rows in groups.items():
        # Her metrik için tohum skorlarını float64 dizilerine topla.
        macro_f1 = np.asarray(
            [row['val_macro_f1'] for row in candidate_rows],
            dtype=np.float64,
        )
        accuracy = np.asarray(
            [row['val_accuracy'] for row in candidate_rows],
            dtype=np.float64,
        )
        balanced_accuracy = np.asarray(
            [row['val_balanced_accuracy'] for row in candidate_rows],
            dtype=np.float64,
        )
        losses = np.asarray(
            [row['val_loss'] for row in candidate_rows],
            dtype=np.float64,
        )
        aggregates.append(
            {
                'corpus': key[0],
                'candidate': key[1],
                'feature_fingerprint': key[2],
                'frame_strategy': key[3],
                'n_mels': key[4],
                'n_frames': key[5],
                'vector_size': key[6],
                'runs': len(candidate_rows),
                'seeds': ','.join(
                    str(seed)
                    for seed in sorted({int(row['seed']) for row in candidate_rows})
                ),
                'val_macro_f1_mean': float(macro_f1.mean()),
                # ddof=0: popülasyon std'si — üç tohum "örneklem" değil,
                # deneyin tamamı olarak özetlenir.
                'val_macro_f1_std': float(macro_f1.std(ddof=0)),
                'val_macro_f1_min': float(macro_f1.min()),
                'val_macro_f1_max': float(macro_f1.max()),
                'val_accuracy_mean': float(accuracy.mean()),
                'val_accuracy_std': float(accuracy.std(ddof=0)),
                'val_balanced_accuracy_mean': float(balanced_accuracy.mean()),
                'val_balanced_accuracy_std': float(
                    balanced_accuracy.std(ddof=0)
                ),
                'val_loss_mean': float(losses.mean()),
            }
        )

    # Sıralama: korpus içinde ortalama macro-F1 azalan; eşitlikte önce düşük
    # std (daha kararlı olan), sonra düşük ortalama kayıp kazanır.
    aggregates.sort(
        key=lambda row: (
            str(row['corpus']),
            -float(row['val_macro_f1_mean']),
            float(row['val_macro_f1_std']),
            float(row['val_loss_mean']),
        )
    )
    # Sıra numarası her korpus için 1'den başlar (korpuslar bağımsız yarışır).
    ranks: dict[str, int] = {}
    for row in aggregates:
        corpus = str(row['corpus'])
        ranks[corpus] = ranks.get(corpus, 0) + 1
        row['rank'] = ranks[corpus]
    return aggregates


def paired_candidate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    '''Ana aday ile rakibin skorlarını BİREBİR AYNI tohum kümesi üzerinde karşılaştırır.

    Eşleştirilmiş (paired) karşılaştırmanın gücü: iki adayı ortalamalar
    yerine tohum tohum kıyaslarız. "Seed 42'de kim kazandı? Seed 143'te?"
    Tohum kaynaklı gürültü her iki adayı da aynı anda etkilediği için,
    tohum başına fark, ortalama farkından çok daha güvenilir bir sinyaldir.
    Çıktıda kazanma sayıları (main_wins / challenger_wins / ties) ve
    ortalama fark raporlanır.
    '''

    # Aday adlarını öneklerine göre ayır ve tam olarak 1'er tane olduğunu
    # doğrula — ikiden fazla aday bu ikili karşılaştırmayı anlamsız kılardı.
    candidates = {str(row['candidate']) for row in rows}
    main_candidates = {
        candidate for candidate in candidates if candidate.startswith('main_')
    }
    challenger_candidates = {
        candidate
        for candidate in candidates
        if candidate.startswith('challenger_')
    }
    if len(main_candidates) != 1 or len(challenger_candidates) != 1:
        raise ValueError('rows must contain one main and one challenger candidate.')
    main_candidate = main_candidates.pop()
    challenger_candidate = challenger_candidates.pop()

    def scores_for(candidate: str) -> dict[int, float]:
        # Adayın {tohum: skor} sözlüğünü kurar; aynı tohum iki kez
        # geçiyorsa veri bozuk demektir, hata ver.
        candidate_rows = [row for row in rows if row['candidate'] == candidate]
        scores = {
            int(row['seed']): float(row['val_macro_f1'])
            for row in candidate_rows
        }
        if len(scores) != len(candidate_rows):
            raise ValueError(f'duplicate seed rows found for {candidate}.')
        return scores

    main_scores = scores_for(main_candidate)
    challenger_scores = scores_for(challenger_candidate)
    # Eşleştirme ancak tohum kümeleri birebir aynıysa geçerlidir.
    if set(main_scores) != set(challenger_scores):
        raise ValueError('main and challenger must use identical seed sets.')

    paired_rows = []
    main_wins = 0
    challenger_wins = 0
    ties = 0
    for seed in sorted(main_scores):
        difference = main_scores[seed] - challenger_scores[seed]
        # Fark makine hassasiyeti kadar küçükse beraberlik say; işaret
        # gürültüsüne "kazandı" dememek için.
        if np.isclose(difference, 0.0, rtol=0.0, atol=1e-12):
            winner = 'tie'
            ties += 1
        elif difference > 0.0:
            winner = 'main'
            main_wins += 1
        else:
            winner = 'challenger'
            challenger_wins += 1
        paired_rows.append(
            {
                'seed': seed,
                'main_val_macro_f1': main_scores[seed],
                'challenger_val_macro_f1': challenger_scores[seed],
                'main_minus_challenger': difference,
                'winner': winner,
            }
        )
    return {
        'main_candidate': main_candidate,
        'challenger_candidate': challenger_candidate,
        'seeds': sorted(main_scores),
        'paired_runs': paired_rows,
        'main_wins': main_wins,
        'challenger_wins': challenger_wins,
        'ties': ties,
        'mean_main_minus_challenger': float(
            np.mean([row['main_minus_challenger'] for row in paired_rows])
        ),
    }


def reference_model(corpus: str) -> MLPConfig:
    # Ablasyonla aynı sabit referans modeli döndürür (tek doğruluk kaynağı
    # feature_ablation.REFERENCE_MODELS); bilinmeyen korpus adını erken yakalar.
    if corpus not in REFERENCE_MODELS:
        raise ValueError(f'Unsupported corpus: {corpus!r}.')
    return REFERENCE_MODELS[corpus]


def _plot_stability(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    path: str | Path,
    title: str,
) -> None:
    # Kararlılık grafiği: her aday için ortalama +- std hata çubuğu (elmas
    # işaretli) ve üzerine tekil tohum skorları (siyah noktalar). Böylece hem
    # özet istatistik hem ham veri aynı görselde okunur.
    import matplotlib

    # 'Agg' arka ucu: ekransız (headless) ortamda da PNG üretebilmek için;
    # pyplot importundan ÖNCE seçilmesi gerekir.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ordered = sorted(aggregates, key=lambda row: int(row['rank']))
    positions = np.arange(len(ordered), dtype=np.float64)
    means = [float(row['val_macro_f1_mean']) for row in ordered]
    errors = [float(row['val_macro_f1_std']) for row in ordered]
    # X ekseni etiketi: "96×64\ncrop-pad" gibi iki satırlık kompakt kimlik.
    labels = [
        (
            f'{int(row["n_mels"])}×{int(row["n_frames"])}\n'
            f'{str(row["frame_strategy"]).replace("_", "-")}'
        )
        for row in ordered
    ]
    colors = ['#2878b5', '#e07a1f']
    fig, axis = plt.subplots(figsize=(7.2, 4.6))

    for index, (position, aggregate) in enumerate(
        zip(positions, ordered, strict=True)
    ):
        # Ortalama +- std hata çubuğu. Legend etiketi yalnızca ilk adaya
        # verilir; yoksa aynı açıklama iki kez listelenirdi.
        axis.errorbar(
            position,
            means[index],
            yerr=errors[index],
            fmt='D',
            markersize=9,
            color=colors[index],
            ecolor=colors[index],
            elinewidth=2.2,
            capsize=8,
            capthick=2.2,
            zorder=2,
            label='Üç-seed ortalaması ± std' if index == 0 else None,
        )
        candidate_rows = [
            row for row in rows if row['candidate'] == aggregate['candidate']
        ]
        candidate_rows.sort(key=lambda row: int(row['seed']))
        # Tekil tohum noktalarını yatayda hafifçe kaydır (jitter): üst üste
        # binip birbirini gizlemesinler.
        offsets = np.linspace(-0.12, 0.12, len(candidate_rows))
        scores = [float(row['val_macro_f1']) for row in candidate_rows]
        axis.scatter(
            position + offsets,
            scores,
            color='#202020',
            s=34,
            zorder=4,
            label='Tekil seed sonucu' if index == 0 else None,
        )
        # Ortalama değeri hata çubuğunun tepesine sayı olarak yaz.
        axis.text(
            position,
            means[index] + errors[index] + 0.006,
            f'{means[index]:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    # Y sınırlarını veriye göre ayarla (alt 0, üst 1 ile sınırlı): farklar
    # görünür olsun ama eksen olasılık aralığından taşmasın.
    all_scores = [float(row['val_macro_f1']) for row in rows]
    lower = max(0.0, min(all_scores) - 0.035)
    upper = min(1.0, max(all_scores) + 0.045)
    axis.set_ylim(lower, upper)
    axis.set_xticks(positions, labels)
    axis.set_ylabel('Validation macro-F1')
    axis.set_title(title)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(loc='lower left')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    # Figürü kapatmak bellek sızıntısını önler (matplotlib figürleri açık
    # kaldıkça birikir).
    plt.close(fig)


def _persist_corpus_rows(
    rows: list[dict[str, Any]],
    corpus_dir: Path,
) -> list[dict[str, Any]]:
    # Mevcut satırların TAMAMINI diske yazar: hem devam-etme dosyası
    # (partial) hem nihai koşu tablosu, hem özet CSV hem grafik. Her deneme
    # sonrasında çağrılır; böylece diskteki durum her an tutarlı ve günceldir.
    ordered = pd.DataFrame(rows).sort_values(['candidate', 'seed'])
    ordered.to_csv(corpus_dir / 'feature_stability.partial.csv', index=False)
    ordered.to_csv(corpus_dir / 'feature_stability_runs.csv', index=False)
    aggregates = aggregate_stability_rows(rows)
    pd.DataFrame(aggregates).to_csv(
        corpus_dir / 'feature_stability.csv',
        index=False,
    )
    corpus = str(rows[0]['corpus'])
    _plot_stability(
        rows,
        aggregates,
        corpus_dir / 'feature_stability.png',
        f'{CORPUS_DISPLAY_NAMES.get(corpus, corpus.upper())} Mel temsil kararlılığı',
    )
    return aggregates


def run_corpus_stability(
    corpus: str,
    *,
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    max_epochs: int,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> dict[str, Any]:
    '''İki sabit öznitelik adayını, test sesi yüklemeden tohumlar arasında eğitir.

    Döngü yapısı bilinçli olarak "önce aday, sonra tohum" şeklindedir:
    bir adayın öznitelik matrisi BİR KEZ yüklenip standardize edilir, sonra
    üç tohumla art arda eğitim yapılır. Ters sırada (önce tohum) her tohum
    için öznitelikler yeniden yüklenirdi — boşuna disk/CPU maliyeti.
    '''

    # --- Girdi doğrulamaları ---
    model_config = reference_model(corpus)
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    # Tohumlar benzersiz ve negatif olmayan olmalı; kopya tohum aynı deneyi
    # iki kez sayar, negatif tohum ise RNG'ler için geçersizdir.
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('seeds must contain unique non-negative integers.')
    if feature_workers < 1 or loader_workers < 0:
        raise ValueError('Worker counts must be feature>=1 and loader>=0.')

    # --- Kurulum: cihaz, klasörler, devam durumu, veri bölmeleri ---
    device = get_device(device_name)
    corpus_dir = ensure_dir(Path(output_root) / corpus)
    history_dir = ensure_dir(corpus_dir / 'histories')
    partial_path = corpus_dir / 'feature_stability.partial.csv'
    candidates = confirmation_candidates(corpus)
    # Ara dosyadan yalnızca bu deney tanımına uyan satırları al; kalanları
    # tamamlanmış işler kümesine çevir.
    rows = _matching_rows(
        _load_partial(partial_path),
        max_epochs=max_epochs,
        model_config=model_config,
        candidates=candidates,
        seeds=seeds,
    )
    completed = _completed_keys(rows, max_epochs, model_config)
    # Test bölmesi bilerek kullanılmadan bırakılır (_held_out_test_frame):
    # bu deney yalnızca eğitim + doğrulama görür.
    train_frame, validation_frame, _held_out_test_frame = _splits_for(
        corpus,
        manifest_path,
    )
    total_trials = len(candidates) * len(seeds)
    trial = 0

    for candidate_name, feature_config in candidates:
        # Bu adayın TÜM tohumları zaten bittiyse öznitelikleri hiç yükleme;
        # sadece bilgi satırlarını bas ve sıradaki adaya geç.
        if all(
            (feature_config.fingerprint, seed) in completed
            for seed in seeds
        ):
            for seed in seeds:
                trial += 1
                print(
                    f'{corpus}: stability trial {trial}/{total_trials} already '
                    f'complete ({candidate_name}, seed={seed})'
                )
            continue

        # Adayın öznitelikleri tohum döngüsünün DIŞINDA bir kez yüklenir.
        cache_dir = Path(cache_root) / corpus
        train_features, train_labels = load_feature_matrix(
            train_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} stability train {candidate_name}',
        )
        validation_features, validation_labels = load_feature_matrix(
            validation_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} stability validation {candidate_name}',
        )
        # Standardizasyon yine yalnızca eğitim istatistikleriyle (sızıntı yok).
        standardizer = FeatureStandardizer.fit(train_features)
        train_features = standardizer.transform(train_features)
        validation_features = standardizer.transform(validation_features)

        for seed in seeds:
            trial += 1
            key = (feature_config.fingerprint, seed)
            # Tek tek tohum bazında da atlama yapılabilir (adayın bir kısmı
            # önceki çalıştırmada bitmiş olabilir).
            if key in completed:
                print(
                    f'{corpus}: stability trial {trial}/{total_trials} already '
                    f'complete ({candidate_name}, seed={seed})'
                )
                continue

            print(
                f'{corpus}: stability trial {trial}/{total_trials} '
                f'({candidate_name}, seed={seed})'
            )
            started = time.perf_counter()
            # Aynı veriler, aynı model, FARKLI tohum: skorlar arasındaki tüm
            # fark yalnızca rastgelelikten (init + batch sırası) gelir.
            outcome = train_with_early_stopping(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                model_config,
                input_dim=feature_config.vector_size,
                num_classes=NUM_CLASSES,
                device=device,
                max_epochs=max_epochs,
                seed=seed,
                num_workers=loader_workers,
                amp=device.type == 'cuda',
            )
            elapsed = time.perf_counter() - started
            metrics = outcome.validation_metrics
            # Sonuç satırı: deneme kimliği + protokol kanıtları + skorlar.
            row = {
                'protocol_version': PROTOCOL_VERSION,
                'corpus': corpus,
                'candidate': candidate_name,
                'feature_fingerprint': feature_config.fingerprint,
                'frame_strategy': feature_config.frame_strategy,
                'n_mels': feature_config.n_mels,
                'n_frames': feature_config.n_frames,
                'vector_size': feature_config.vector_size,
                'seed': seed,
                'max_epochs': max_epochs,
                'model_config': json.dumps(
                    model_config.to_dict(),
                    sort_keys=True,
                ),
                'class_weighting': 'inverse-frequency CrossEntropyLoss',
                'best_epoch': outcome.best_epoch,
                'epochs_trained': outcome.epochs_trained,
                'stopped_early': outcome.stopped_early,
                'val_loss': outcome.validation_loss,
                'val_accuracy': metrics['accuracy'],
                'val_balanced_accuracy': metrics['balanced_accuracy'],
                'val_macro_f1': metrics['macro_f1'],
                'val_weighted_f1': metrics['weighted_f1'],
                'elapsed_seconds': elapsed,
                'test_features_loaded': False,
            }
            rows.append(row)
            completed.add(key)
            # Epoch-epoch eğitim geçmişi ayrı dosyaya (öğrenme eğrileri için).
            pd.DataFrame(outcome.history).to_csv(
                history_dir
                / f'{candidate_name}_seed_{seed}_{feature_config.fingerprint}.csv',
                index=False,
            )
            # Her denemeden sonra tüm çıktıları güncelle: kesinti olursa
            # hiçbir tamamlanmış sonuç kaybolmaz.
            _persist_corpus_rows(rows, corpus_dir)
            print(
                f'{corpus}: {candidate_name} seed={seed} validation '
                f'macro-F1={metrics["macro_f1"]:.4f}'
            )
            del outcome

        # Adayın büyük matrislerini bırak: sıradaki adayın yüklemesi
        # sırasında bellek iki katına çıkmasın.
        del train_features, validation_features

    # Son kez topla ve eşleştirilmiş karşılaştırmayı JSON olarak yaz.
    aggregates = _persist_corpus_rows(rows, corpus_dir)
    comparison = paired_candidate_comparison(rows)
    (corpus_dir / 'feature_stability_comparison.json').write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding='utf-8',
    )
    return {
        'runs': rows,
        'aggregates': aggregates,
        'comparison': comparison,
    }


def _json_default(value: Any) -> Any:
    # json.dumps numpy skalerlerini ve Path nesnelerini seri hale getiremez;
    # bu geri çağırma (default hook) onları düz Python tiplerine çevirir.
    # Tanınmayan tipler için TypeError fırlatmak json sözleşmesinin gereğidir.
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def run_feature_stability(
    *,
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    manifest_path: str | Path = 'odev1/manifest_subset.csv',
    cache_root: str | Path = 'data/cache/odev3_melspec',
    output_root: str | Path = 'odev3/feature_stability',
    max_epochs: int = 60,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> dict[str, dict[str, Any]]:
    # Tüm korpuslar için kararlılık deneyini çalıştıran ve tek bir
    # summary.json'da toplayan çatı fonksiyon. Özet, deneyin tam tanımını
    # (protokol, seçim kuralı, tohumlar, modeller, adaylar) içerir — rapor
    # yazarken tek başvuru kaynağı budur.
    results = {
        corpus: run_corpus_stability(
            corpus,
            manifest_path=manifest_path,
            cache_root=cache_root,
            output_root=output_root,
            max_epochs=max_epochs,
            seeds=seeds,
            device_name=device_name,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
        )
        for corpus in corpora
    }
    summary = {
        'protocol_version': PROTOCOL_VERSION,
        'protocol': (
            'validation only; two fixed Mel candidates per corpus; '
            'held-out test audio is not loaded'
        ),
        'selection_rule': 'highest mean validation macro-F1 across fixed seeds',
        'class_weighting': 'inverse-frequency CrossEntropyLoss',
        'test_features_loaded': False,
        'seeds': list(seeds),
        'max_epochs': max_epochs,
        'reference_models': {
            corpus: reference_model(corpus).to_dict() for corpus in corpora
        },
        'candidates': {
            corpus: [
                {'name': name, 'config': asdict(config)}
                for name, config in confirmation_candidates(corpus)
            ]
            for corpus in corpora
        },
        'per_dataset': results,
    }
    output_root = ensure_dir(output_root)
    (Path(output_root) / 'summary.json').write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding='utf-8',
    )
    return results


def main() -> None:
    # Bağımsız CLI: kararlılık deneyi ablasyondan ayrı çalıştırılabilir.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--corpora',
        nargs='+',
        default=['cremad', 'meld'],
        choices=sorted(REFERENCE_MODELS),
    )
    parser.add_argument('--manifest', default='odev1/manifest_subset.csv')
    parser.add_argument('--cache-root', default='data/cache/odev3_melspec')
    parser.add_argument('--out-root', default='odev3/feature_stability')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIRMATION_SEEDS)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=1)
    parser.add_argument('--loader-workers', type=int, default=0)
    args = parser.parse_args()
    try:
        run_feature_stability(
            corpora=tuple(args.corpora),
            manifest_path=args.manifest,
            cache_root=args.cache_root,
            output_root=args.out_root,
            max_epochs=args.max_epochs,
            seeds=tuple(args.seeds),
            device_name=args.device,
            feature_workers=args.feature_workers,
            loader_workers=args.loader_workers,
        )
    except ValueError as error:
        # Doğrulama hatalarını CLI diline çevir: kullanım bilgisiyle çık.
        parser.error(str(error))


if __name__ == '__main__':
    main()
