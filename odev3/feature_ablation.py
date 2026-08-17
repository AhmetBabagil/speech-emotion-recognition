'''Sabit boyutlu Mel zaman temsillerinin YALNIZCA doğrulama üzerinde karşılaştırılması.

"Ablasyon" (ablation) çalışması, bir sistemin bileşenlerini tek tek
değiştirip/etkisizleştirip performansa etkisini ölçmek demektir. Burada
model mimarisini SABİT tutup ÖZNİTELİK tarafını değiştiriyoruz: kaç mel
bandı, kaç zaman karesi, hangi zaman-sabitleme stratejisi (crop_pad mı
resize mı) daha iyi çalışıyor?

En kritik protokol kuralı: test kümesine ait ses dosyaları bu deneyde HİÇ
YÜKLENMEZ. Bütün karşılaştırma eğitim + doğrulama katmanlarında yapılır;
böylece öznitelik seçimi test kümesine "bakmış" olmaz ve nihai test skoru
tarafsız kalır. (Çıktıdaki test_features_loaded=False alanı bunu belgeler.)
'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

# Depo kökünü import yoluna ekle; dosya doğrudan çalıştırıldığında da
# odev3/ ve ser/ paketleri bulunabilsin.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev3.dataset import FeatureStandardizer, load_feature_matrix
from odev3.features_melspec import MelSpecConfig
from odev3.model import MLPConfig
from odev3.pipeline import SEED, _splits_for
from odev3.training import train_with_early_stopping
from ser.constants import NUM_CLASSES
from ser.utils import ensure_dir, get_device


# Her korpus için SABİT referans model: ana hiperparametre aramasının o
# korpustaki kazananı. Modeli sabitlemek şart — aynı anda hem model hem
# öznitelik değişseydi hangi farkın neden kaynaklandığı anlaşılamazdı
# (kontrollü deney ilkesi: tek seferde tek değişken).
REFERENCE_MODELS = {
    'cremad': MLPConfig(
        batch_size=64,
        learning_rate=3e-4,
        patience=12,
        hidden_dims=(768, 384),
        activation='relu',
        batch_norm=True,
        dropout=0.3,
        weight_decay=1e-4,
    ),
    'meld': MLPConfig(
        batch_size=64,
        learning_rate=3e-4,
        patience=8,
        hidden_dims=(512, 256),
        activation='relu',
        batch_norm=True,
        dropout=0.5,
        weight_decay=1e-4,
    ),
}


def feature_candidates() -> tuple[MelSpecConfig, ...]:
    '''Zaman işleme ve frekans çözünürlüğünü kontrollü çiftler halinde karşılaştırır.

    Adaylar bilinçli olarak ÇİFTLER halinde kurulur: her (n_mels, n_frames)
    kombinasyonu hem crop_pad hem resize stratejisiyle denenir. Böylece
    "strateji farkı" her çözünürlükte ayrı ayrı gözlemlenebilir. Toplam
    8 aday: 2 kare sayısı x 2 strateji + 2 ek mel çözünürlüğü x 2 strateji.
    '''

    return (
        MelSpecConfig(n_frames=64, frame_strategy='crop_pad'),
        MelSpecConfig(n_frames=64, frame_strategy='resize'),
        MelSpecConfig(n_frames=96, frame_strategy='crop_pad'),
        MelSpecConfig(n_frames=96, frame_strategy='resize'),
        MelSpecConfig(n_mels=80, n_frames=64, frame_strategy='crop_pad'),
        MelSpecConfig(n_mels=80, n_frames=64, frame_strategy='resize'),
        MelSpecConfig(n_mels=96, n_frames=64, frame_strategy='crop_pad'),
        MelSpecConfig(n_mels=96, n_frames=64, frame_strategy='resize'),
    )


def _normalized_model_config(payload: Any) -> str | None:
    # Model konfigürasyonunu karşılaştırılabilir tek biçimli JSON metnine
    # çevirir. CSV'den okunan değer string, bellekteki değer sözlük olabilir;
    # ikisini de aynı kanonik biçime (sıralı anahtarlar, boşluksuz) indirger.
    # Çözümlemesi mümkün olmayan değerler None döner ve eşleşme sayılmaz.
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_keys(
    rows: list[dict[str, Any]],
    max_epochs: int,
    model_config: MLPConfig,
) -> set[str]:
    # Devam etme (resume) mantığının kalbi: yarım kalan CSV'deki satırlardan,
    # ŞU ANKİ ayarlarla (aynı seed, aynı epoch limiti, aynı referans model)
    # üretilmiş olanların öznitelik parmak izlerini toplar. Ayarlar farklıysa
    # eski satır "tamamlanmış" sayılmaz — bayat sonuç yeniden üretilir.
    expected_model = _normalized_model_config(model_config.to_dict())
    return {
        str(row['feature_fingerprint'])
        for row in rows
        if int(row['seed']) == SEED and int(row['max_epochs']) == max_epochs
        and _normalized_model_config(row.get('model_config')) == expected_model
    }


def _load_partial(path: Path) -> list[dict[str, Any]]:
    # Önceki (yarıda kesilmiş) çalıştırmanın ara CSV'sini okur; dosya yoksa
    # sıfırdan başlanır. Bu sayede uzun süren ablasyon, elektrik kesintisi
    # gibi durumlarda kaldığı yerden devam edebilir.
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient='records')


def run_corpus_ablation(
    corpus: str,
    *,
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    max_epochs: int,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> list[dict[str, Any]]:
    '''Eşleştirilmiş öznitelik adaylarını yalnızca eğitim ve doğrulama katmanlarıyla eğitir.

    Tek korpus için akış: her öznitelik adayında (1) öznitelik matrislerini
    yükle/önbellekten getir, (2) eğitim istatistikleriyle standardize et,
    (3) sabit referans modeli erken durdurmayla eğit, (4) doğrulama
    skorlarını CSV'ye ekle. Sonunda adaylar macro-F1'e göre sıralanır.
    '''

    if corpus not in REFERENCE_MODELS:
        raise ValueError(f'Unsupported corpus: {corpus!r}.')
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')

    # --- Kurulum: cihaz, çıktı klasörleri, devam durumu, veri bölmeleri ---
    device = get_device(device_name)
    corpus_dir = ensure_dir(Path(output_root) / corpus)
    history_dir = ensure_dir(corpus_dir / 'histories')
    partial_path = corpus_dir / 'feature_ablation.partial.csv'
    model_config = REFERENCE_MODELS[corpus]
    rows = _load_partial(partial_path)
    completed = _completed_keys(rows, max_epochs, model_config)
    # _splits_for testi de döndürür ama burada değişken adı bilerek
    # _held_out_test_frame: test SADECE alınır, asla kullanılıp yüklenmez.
    train_frame, validation_frame, _held_out_test_frame = _splits_for(
        corpus,
        manifest_path,
    )
    candidates = feature_candidates()
    total_trials = len(candidates)

    for trial, feature_config in enumerate(candidates, start=1):
        # Daha önce tamamlanmış aday: eğitimi atla, sadece bilgi ver.
        if feature_config.fingerprint in completed:
            print(
                f'{corpus}: feature trial {trial}/{total_trials} already complete '
                f'({feature_config.n_mels} mels, {feature_config.n_frames} frames, '
                f'{feature_config.frame_strategy})'
            )
            continue

        print(
            f'{corpus}: feature trial {trial}/{total_trials} '
            f'({feature_config.n_mels} mels, {feature_config.n_frames} frames, '
            f'{feature_config.frame_strategy})'
        )
        started = time.perf_counter()
        cache_dir = Path(cache_root) / corpus
        # Öznitelikleri yükle: cache varsa hızlı, yoksa çıkarılıp cache'lenir.
        # Her aday farklı fingerprint'e sahip olduğundan cache'ler karışmaz.
        train_features, train_labels = load_feature_matrix(
            train_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} ablation train {trial}',
        )
        validation_features, validation_labels = load_feature_matrix(
            validation_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} ablation validation {trial}',
        )
        # Standardizasyon SADECE eğitim istatistikleriyle öğrenilir; doğrulama
        # yalnızca transform edilir (veri sızıntısını önleme kuralı).
        standardizer = FeatureStandardizer.fit(train_features)
        train_features = standardizer.transform(train_features)
        validation_features = standardizer.transform(validation_features)
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
            seed=SEED,
            num_workers=loader_workers,
            amp=device.type == 'cuda',
        )
        elapsed = time.perf_counter() - started
        metrics = outcome.validation_metrics
        # Sonuç satırı: adayın kimliği + eğitim özeti + doğrulama skorları.
        # test_features_loaded=False, protokolün kanıtı olarak açıkça yazılır.
        row = {
            'corpus': corpus,
            'trial': trial,
            'seed': SEED,
            'max_epochs': max_epochs,
            'feature_fingerprint': feature_config.fingerprint,
            'frame_strategy': feature_config.frame_strategy,
            'n_mels': feature_config.n_mels,
            'n_frames': feature_config.n_frames,
            'vector_size': feature_config.vector_size,
            'model_config': json.dumps(model_config.to_dict(), sort_keys=True),
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
        completed.add(feature_config.fingerprint)
        # Her denemenin epoch-epoch öğrenme geçmişi ayrı dosyaya yazılır
        # (öğrenme eğrisi çizimleri için).
        pd.DataFrame(outcome.history).to_csv(
            history_dir / f'trial_{trial:02d}_{feature_config.fingerprint}.csv',
            index=False,
        )
        # Ara CSV her denemeden sonra güncellenir: çalışma kesilirse bir
        # sonraki başlatma tam bu noktadan devam eder.
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(
            f'{corpus}: trial {trial} validation macro-F1='
            f'{metrics["macro_f1"]:.4f}'
        )

        # Büyük matrisleri ve modeli açıkça bırak: bir sonraki adayın
        # öznitelikleri yüklenirken bellek tepe noktasını düşürür.
        del train_features, validation_features, outcome

    # Nihai tablo: macro-F1'e göre azalan sıralama; eşitlikte dengeli doğruluk
    # (yüksek iyi) ve doğrulama kaybı (düşük iyi) beraberliği bozar.
    ranked = pd.DataFrame(rows).sort_values(
        ['val_macro_f1', 'val_balanced_accuracy', 'val_loss'],
        ascending=[False, False, True],
    )
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    ranked.to_csv(corpus_dir / 'feature_ablation.csv', index=False)
    return ranked.to_dict(orient='records')


def run_feature_ablation(
    *,
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    manifest_path: str | Path = 'odev1/manifest_subset.csv',
    cache_root: str | Path = 'data/cache/odev3_melspec',
    output_root: str | Path = 'odev3/feature_ablation',
    max_epochs: int = 60,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    # Tüm korpuslar için ablasyonu çalıştırıp tek bir summary.json üreten
    # çatı fonksiyon. Özet dosyasına protokol beyanı, seed, referans modeller
    # ve tüm aday tanımları yazılır — rapor bu dosyadan beslenir.
    results = {
        corpus: run_corpus_ablation(
            corpus,
            manifest_path=manifest_path,
            cache_root=cache_root,
            output_root=output_root,
            max_epochs=max_epochs,
            device_name=device_name,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
        )
        for corpus in corpora
    }
    summary = {
        'protocol': 'validation only; held-out test audio is not loaded',
        'seed': SEED,
        'max_epochs': max_epochs,
        'reference_models': {
            corpus: config.to_dict() for corpus, config in REFERENCE_MODELS.items()
        },
        'feature_candidates': [asdict(config) for config in feature_candidates()],
        'results': results,
    }
    output_root = ensure_dir(output_root)
    (Path(output_root) / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return results


def main() -> None:
    # Bağımsız CLI: ablasyon, ana deneyden ayrı olarak da çalıştırılabilir.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--corpora',
        nargs='+',
        default=['cremad', 'meld'],
        choices=sorted(REFERENCE_MODELS),
    )
    parser.add_argument('--manifest', default='odev1/manifest_subset.csv')
    parser.add_argument('--cache-root', default='data/cache/odev3_melspec')
    parser.add_argument('--out-root', default='odev3/feature_ablation')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=1)
    parser.add_argument('--loader-workers', type=int, default=0)
    args = parser.parse_args()
    if args.feature_workers < 1 or args.loader_workers < 0:
        parser.error('Worker counts must be feature>=1 and loader>=0.')
    run_feature_ablation(
        corpora=tuple(args.corpora),
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        output_root=args.out_root,
        max_epochs=args.max_epochs,
        device_name=args.device,
        feature_workers=args.feature_workers,
        loader_workers=args.loader_workers,
    )


if __name__ == '__main__':
    main()
