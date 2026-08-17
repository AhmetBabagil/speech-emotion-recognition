'''Tek corpus için uçtan uca final deney hattı.

Akış (her yöntem için ayrı ayrı):
1. Manifest'i oku, konuşmacı-bağımsız eğitim/geçerleme/test bölmesi yap.
2. Adayların özniteliklerini çıkar (önbellekten geliyorsa saniyeler sürer).
3. ARAMA: her adayı eğit, geçerleme macro-F1'ine göre sırala.
4. İYİLEŞTİRME TURU: kazananın çevresindeki yerel adayları da dene.
5. Nihai kazananı TEST kümesinde BİR KEZ değerlendir.
6. Tüm artefaktları (arama logu, kazanan ayarlar, öğrenme eğrisi, metrikler,
   karışıklık matrisi, model ağırlıkları) çıktı klasörüne yaz.

Dürüstlük ilkesi: test kümesine yalnızca seçilmiş nihai model dokunur;
hiperparametre kararlarının hiçbiri test sonucuna bakılarak verilmez.
'''

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from final.dataset import Standardizer, load_feature_tensor
from final.features import (
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import MelCNN, RNNConfig, SeqRNN, count_parameters
from final.search_space import (
    cnn_refinement,
    cnn_space,
    rnn_refinement,
    rnn_space,
)
from final.training import (
    TrainingOutcome,
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.data.splits import prepare_splits
from ser.constants import NUM_CLASSES
from ser.evaluate import report as evaluate_report
from ser.utils import ensure_dir, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SplitSettings:
    '''ser.config'in veri bölümünün, prepare_splits'e yetecek kadarı.

    split="speaker": bölme aktör kimliğine göre yapılır — bir konuşmacının
    tüm kayıtları tek katmanda kalır (konuşmacı-bağımsız protokol).
    '''

    train_corpora: tuple[str, ...]
    eval_corpora: tuple[str, ...]
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split: str = 'speaker'


def _limit_stratified(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    '''SADECE TANI amaçlı: katmanı, sınıf oranlarını koruyarak küçültür.

    Duman testlerinde (quick mod) tüm hattı dakikalar içinde uçtan uca
    denemek için kullanılır; gerçek deneylerde devrede değildir.
    '''

    if limit >= len(df):
        return df
    parts = []
    rng_seed = seed
    for _, group in df.groupby('label_idx'):
        # Her sınıftan, o sınıfın genel orandaki payı kadar örnek al.
        take = max(1, int(round(limit * len(group) / len(df))))
        parts.append(group.sample(n=min(take, len(group)), random_state=rng_seed))
        rng_seed += 1
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _feature_folds(
    folds: dict[str, pd.DataFrame],
    feature_cfg,
    extract_fn: Callable,
    cache_root: str | Path,
    *,
    workers: int,
    cache: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    '''Verilen katmanların öznitelik tensörlerini üretir (veya yeniden kullanır).

    ``cache`` sözlüğü koşu-içi bellektir: aynı öznitelik ayarını kullanan
    birden çok aday, katman tensörlerini yalnızca bir kez hesaplatır.
    '''

    result = {}
    for name, records in folds.items():
        key = (feature_cfg.fingerprint, name)
        if key not in cache:
            cache[key] = load_feature_tensor(
                records,
                cache_root,
                feature_cfg.fingerprint,
                lambda path: extract_fn(path, feature_cfg),
                feature_cfg.shape,
                workers=workers,
                description=f'{feature_cfg.fingerprint} {name}',
            )
        result[name] = cache[key]
    return result


def _candidate_row(feature_cfg, model_cfg, outcome: TrainingOutcome, seconds: float,
                   stage: str, n_params: int) -> dict[str, Any]:
    '''Bir adayın tüm sonuçlarını arama loguna yazılacak tek satıra çevirir.'''

    row = {
        'stage': stage,                                              # 'search' / 'refine'
        'feature_fingerprint': feature_cfg.fingerprint,
        'feature_config': json.dumps(feature_cfg.__dict__, sort_keys=True),
        'model_config': json.dumps(model_cfg.to_dict(), sort_keys=True),
        'parameters': n_params,
        'best_epoch': outcome.best_epoch,
        'epochs_trained': outcome.epochs_trained,
        'stopped_early': outcome.stopped_early,
        'val_loss': outcome.validation_loss,
        'val_accuracy': outcome.validation_metrics['accuracy'],
        'val_balanced_accuracy': outcome.validation_metrics['balanced_accuracy'],
        'val_macro_f1': outcome.validation_metrics['macro_f1'],
        'val_weighted_f1': outcome.validation_metrics['weighted_f1'],
        'seconds': round(seconds, 1),
    }
    return row


def _plot_history(history: list[dict[str, Any]], out_path: Path, title: str) -> None:
    '''Kazanan adayın öğrenme eğrilerini (loss + macro-F1) PNG olarak kaydeder.

    Bu grafik, erken durdurmanın çalıştığının görsel kanıtıdır: eğitim
    kaybı düşmeye devam ederken geçerleme kaybının dönmesi = aşırı öğrenme.
    '''

    import matplotlib
    matplotlib.use('Agg')   # ekran gerektirmeyen arka uç (sunucu/terminal uyumlu)
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h['train_loss'] for h in history], label='train')
    ax1.plot(epochs, [h['val_loss'] for h in history], label='validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted CE loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h['train_macro_f1'] for h in history], label='train')
    ax2.plot(epochs, [h['val_macro_f1'] for h in history], label='validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_method(
    method: str,
    folds: dict[str, pd.DataFrame],
    *,
    cache_root: Path,
    out_dir: Path,
    grid_mode: str,
    max_epochs: int,
    device: torch.device,
    feature_workers: int,
    loader_workers: int,
    amp: bool,
    refine: bool,
    seed: int,
) -> dict[str, Any]:
    '''Tek bir yöntemi arar, iyileştirir ve test eder; özet sözlüğü döndürür.'''

    # Yönteme göre aday listesi, öznitelik fonksiyonu ve normalizasyon ekseni seç.
    if method == 'cnn':
        candidates = cnn_space(grid_mode)
        extract_fn = extract_mel_image
        feature_axis = 1  # [N, mels, T] -> mel bandı başına istatistik
        refinement_fn = cnn_refinement
    elif method == 'rnn':
        candidates = rnn_space(grid_mode)
        extract_fn = extract_interval_series
        feature_axis = 2  # [N, T, D] -> öznitelik boyutu başına istatistik
        refinement_fn = rnn_refinement
    else:
        raise ValueError(f'Bilinmeyen yöntem {method!r}.')

    def build_model(feature_cfg, model_cfg):
        '''Adayın ayarlarından taze (rastgele başlatılmış) bir model kurar.'''

        if method == 'cnn':
            return MelCNN(NUM_CLASSES, model_cfg)
        return SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)

    ensure_dir(out_dir)
    feature_cache: dict = {}                 # koşu-içi öznitelik belleği
    rows: list[dict[str, Any]] = []          # arama logunun satırları
    best: dict[str, Any] | None = None       # şu ana kadarki geçerleme kazananı

    def run_stage(stage: str, stage_candidates) -> None:
        '''Bir aday listesini eğitir; geçerleme kazananını `best`te günceller.'''

        nonlocal best
        for index, (feature_cfg, model_cfg) in enumerate(stage_candidates, start=1):
            # 1) Bu adayın öznitelikleri (eğitim + geçerleme katmanları).
            tensors = _feature_folds(
                {'train': folds['train'], 'val': folds['val']},
                feature_cfg,
                extract_fn,
                cache_root,
                workers=feature_workers,
                cache=feature_cache,
            )
            train_x, train_y = tensors['train']
            val_x, val_y = tensors['val']
            # 2) Normalizasyon YALNIZ eğitim istatistikleriyle öğrenilir.
            standardizer = Standardizer.fit(train_x, feature_axis)
            model = build_model(feature_cfg, model_cfg)
            n_params = count_parameters(model)
            started = time.perf_counter()
            # 3) Ağırlıklı loss + early stopping ile eğit.
            outcome = train_with_early_stopping(
                model,
                standardizer.transform(train_x),
                train_y,
                standardizer.transform(val_x),
                val_y,
                model_cfg.optim,
                num_classes=NUM_CLASSES,
                device=device,
                max_epochs=max_epochs,
                seed=seed,
                num_workers=loader_workers,
                amp=amp,
            )
            seconds = time.perf_counter() - started
            # 4) Sonucu logla ve gerekirse kazananı güncelle.
            row = _candidate_row(feature_cfg, model_cfg, outcome, seconds, stage, n_params)
            rows.append(row)
            log.info(
                '[%s/%s %d/%d] val macro-F1=%.4f acc=%.4f (epoch %d/%d, %.0fs)',
                method, stage, index, len(stage_candidates),
                row['val_macro_f1'], row['val_accuracy'],
                outcome.best_epoch, outcome.epochs_trained, seconds,
            )
            if best is None or row['val_macro_f1'] > best['row']['val_macro_f1']:
                best = {
                    'row': row,
                    'feature_cfg': feature_cfg,
                    'model_cfg': model_cfg,
                    'standardizer': standardizer,
                    'outcome': outcome,
                }

    # ---- Aşama 1: geniş arama ----
    run_stage('search', candidates)
    # ---- Aşama 2: kazananın çevresinde yerel iyileştirme ----
    if refine and grid_mode != 'quick' and best is not None:
        # Daha önce denenen adaylar tekrar eğitilmesin.
        seen = set(candidates) | {(best['feature_cfg'], best['model_cfg'])}
        refinement = [
            c for c in refinement_fn((best['feature_cfg'], best['model_cfg']))
            if c not in seen
        ]
        if refinement:
            run_stage('refine', refinement)

    if best is None:
        raise RuntimeError(f'{method} için başarılı aday yok.')

    # Arama logunu diske yaz (rapor/sunum tabloları buradan üretilir).
    search_log = pd.DataFrame(rows)
    search_log.to_csv(out_dir / 'search_log.csv', index=False)

    feature_cfg = best['feature_cfg']
    model_cfg = best['model_cfg']
    standardizer = best['standardizer']
    outcome: TrainingOutcome = best['outcome']

    # Kazananın öğrenme geçmişi + eğrisi + ayarları.
    pd.DataFrame(outcome.history).to_csv(out_dir / 'winner_history.csv', index=False)
    _plot_history(outcome.history, out_dir / 'winner_learning_curve.png',
                  f'{method.upper()} winner learning curve')
    with open(out_dir / 'winner.json', 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'method': method,
                'feature_config': feature_cfg.__dict__,
                'model_config': model_cfg.to_dict(),
                'parameters': best['row']['parameters'],
                'best_epoch': outcome.best_epoch,
                'val_metrics': outcome.validation_metrics,
            },
            handle,
            indent=2,
        )
    # Model + normalizasyon parametreleri birlikte kaydedilir; böylece
    # tahmin/demoda birebir aynı ön işleme uygulanabilir.
    torch.save(
        {
            'state_dict': outcome.model.state_dict(),
            'feature_config': feature_cfg.__dict__,
            'model_config': model_cfg.to_dict(),
            'standardizer_mean': standardizer.mean,
            'standardizer_scale': standardizer.scale,
            'feature_axis': standardizer.feature_axis,
        },
        out_dir / 'winner_model.pt',
    )

    # TEST katmanına tam olarak bir kez, yalnızca nihai kazanan dokunur.
    test_x, test_y = _feature_folds(
        {'test': folds['test']},
        feature_cfg,
        extract_fn,
        cache_root,
        workers=feature_workers,
        cache=feature_cache,
    )['test']
    class_weights = inverse_frequency_weights(
        folds['train']['label_idx'].to_numpy(), NUM_CLASSES
    )
    test_loss, _, test_prob = evaluate_arrays(
        outcome.model,
        standardizer.transform(test_x),
        test_y,
        class_weights=class_weights,
        device=device,
        num_workers=loader_workers,
    )
    test_pred = test_prob.argmax(axis=1)
    # evaluate_report: metrics.json + karışıklık matrisi PNG'sini yazar.
    test_metrics = evaluate_report(
        test_y, test_pred, out_dir, prefix='test',
        title=f'{method.upper()} test confusion matrix',
    )
    val_metrics = outcome.validation_metrics
    log.info('[%s] TEST acc=%.4f macro-F1=%.4f (val macro-F1=%.4f)',
             method, test_metrics['accuracy'], test_metrics['macro_f1'],
             val_metrics['macro_f1'])
    return {
        'method': method,
        'winner_feature': feature_cfg.__dict__,
        'winner_model': model_cfg.to_dict(),
        'val': val_metrics,
        'test': test_metrics,
        'test_loss': test_loss,
        'search_rows': len(rows),
    }


def run_all(
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    *,
    corpus: str = 'cremad',
    methods: tuple[str, ...] = ('cnn', 'rnn'),
    grid_mode: str = 'report',
    max_epochs: int = 60,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
    amp: bool = True,
    refine: bool = True,
    limit_per_split: int | None = None,
    prior_results_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    '''Tüm deneyi koşturur: bölme -> her yöntem -> karşılaştırma tablosu.'''

    # 1) Manifest + konuşmacı-bağımsız bölme (deterministik: seed=42).
    manifest = pd.read_csv(manifest_path)
    settings = SplitSettings(train_corpora=(corpus,), eval_corpora=(corpus,))
    train_df, val_df, test_df = prepare_splits(manifest, settings, seed=seed)
    if limit_per_split is not None:
        # Yalnız duman testi: katmanları oransal küçült.
        train_df = _limit_stratified(train_df, limit_per_split, seed)
        val_df = _limit_stratified(val_df, limit_per_split, seed + 1)
        test_df = _limit_stratified(test_df, limit_per_split, seed + 2)
        log.warning('Diagnostic limit active: train=%d val=%d test=%d',
                    len(train_df), len(val_df), len(test_df))
    folds = {'train': train_df, 'val': val_df, 'test': test_df}

    # 2) Cihaz seçimi: varsa GPU, yoksa CPU.
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    log.info('Corpus=%s device=%s grid=%s | train=%d val=%d test=%d',
             corpus, device, grid_mode, len(train_df), len(val_df), len(test_df))

    # 3) Her yöntemi sırayla koştur; çıktılar corpus/yöntem klasörlerine gider.
    output_root = ensure_dir(Path(output_root) / corpus)
    cache_root = Path(cache_root)
    results: dict[str, dict[str, Any]] = {}
    for method in methods:
        results[method] = run_method(
            method,
            folds,
            cache_root=cache_root,
            out_dir=Path(output_root) / method,
            grid_mode=grid_mode,
            max_epochs=max_epochs,
            device=device,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
            amp=amp,
            refine=refine,
            seed=seed,
        )

    # 4) Yöntemleri (ve varsa önceki aşama sonuçlarını) tek tabloda birleştir.
    comparison = _comparison_table(results, prior_results_path)
    comparison.to_csv(Path(output_root) / 'method_comparison.csv', index=False)
    with open(Path(output_root) / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, default=str)
    return results


def _comparison_table(
    results: dict[str, dict[str, Any]],
    prior_results_path: str | Path | None,
) -> pd.DataFrame:
    '''Yöntem karşılaştırma CSV'sini kurar; istenirse eski sonuçları da ekler.'''

    rows = []
    names = {'cnn': 'Yöntem 1: Mel + CNN', 'rnn': 'Yöntem 2: Aralık + LSTM/GRU'}
    for method, result in results.items():
        rows.append({
            'model': names.get(method, method),
            'source': 'final',
            'val_macro_f1': result['val']['macro_f1'],
            'test_accuracy': result['test']['accuracy'],
            'test_balanced_accuracy': result['test']['balanced_accuracy'],
            'test_macro_f1': result['test']['macro_f1'],
            'test_weighted_f1': result['test']['weighted_f1'],
        })
    table = pd.DataFrame(rows)
    if prior_results_path and Path(prior_results_path).is_file():
        try:
            prior = pd.read_csv(prior_results_path)
            prior['source'] = Path(prior_results_path).parent.name
            table = pd.concat([table, prior], ignore_index=True)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            log.warning('Önceki sonuçlar birleştirilemedi: %s', error)
    return table
