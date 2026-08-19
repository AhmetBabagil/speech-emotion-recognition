# Geliştirme aşaması: kazananların artırma / dikkat varyantlarını dener.
#
# Yönergenin 5. bölümü "hiperparametre optimizasyonundan SONRA performans iyileştirmek için geliştirmeler yapın" der. Bu betik tam olarak onu yapar:
#
# 1. run_experiment.py'nin ürettiği kazananları (winner.json) okur.
# 2. Literatürün önerdiği geliştirmelerle yeniden eğitir: CNN -> SpecAugment maskeleme (2 şiddet); RNN -> dikkat havuzlama, öznitelik gürültüsü, ikisi birden.
# 3. Varyantları GEÇERLEME kümesinde tabanla karşılaştırır.
# 4. Yalnızca geçerlemede kazananı geçen varyant test kümesinde değerlendirilir; hiçbiri geçemezse test'e dokunulmaz (dürüst protokol) ve bu olumsuz sonuç da rapora yazılır.
#
# Örnekler:
# python final/improve.py
# python final/improve.py --methods rnn --out-root final/outputs

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
from dataclasses import replace  # ayarın bir alanını değiştirip kopyalamak için
import json  # winner.json okuma + özet yazma
from pathlib import Path  # dosya yolları
import sys  # import yolu
import time  # süre ölçümü
from typing import Any  # tip ipucu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

import pandas as pd  # noqa: E402  # manifest + tablo
import torch  # noqa: E402  # cihaz + model kaydı

from final.augment import FeatureNoise, SpecMask  # noqa: E402  # veri artırmaları
from final.dataset import Standardizer  # noqa: E402  # normalizasyon
from final.features import (  # noqa: E402  # öznitelik ayarları + çıkarıcılar
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import (  # noqa: E402  # modeller + ayarlar
    CNNConfig,
    MelCNN,
    OptimSettings,
    RNNConfig,
    SeqRNN,
)
from final.pipeline import SplitSettings, _feature_folds, _limit_stratified  # noqa: E402  # bölme + yükleme + küçültme
from final.training import (  # noqa: E402  # eğitim + değerlendirme
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402  # sınıf sayısı
from ser.data.splits import prepare_splits  # noqa: E402  # konuşmacı-bağımsız bölme
from ser.evaluate import report as evaluate_report  # noqa: E402  # metrik + matris
from ser.utils import ensure_dir, get_logger  # noqa: E402  # klasör + günlük

log = get_logger(__name__)  # günlükleyici


def _model_config_from_dict(method: str, values: dict[str, Any]):  # winner.json'daki sözlükten config nesnesini geri kurar.

    values = dict(values)  # sözlüğün kopyası
    optim = OptimSettings(**values.pop('optim'))  # optimizasyon ayarını çıkar/kur
    if method == 'cnn':  # CNN ise
        return CNNConfig(channels=tuple(values['channels']),  # CNN ayarı
                         dropout=values['dropout'], optim=optim)
    return RNNConfig(  # RNN ise RNN ayarı
        rnn_type=values['rnn_type'],
        hidden_size=values['hidden_size'],
        num_layers=values['num_layers'],
        bidirectional=values['bidirectional'],
        dropout=values['dropout'],
        pooling=values['pooling'],
        optim=optim,
    )


def _variants(method: str, model_cfg) -> list[dict[str, Any]]:
    # Denenecek geliştirme varyantları; transform=None demek "yalnız model değişikliği, veri artırma yok" demektir.

    if method == 'cnn':  # CNN geliştirmeleri
        return [
            {'name': 'specaugment', 'model_cfg': model_cfg,  # standart SpecAugment
             'transform': SpecMask()},
            {'name': 'specaugment_light', 'model_cfg': model_cfg,  # hafif SpecAugment
             'transform': SpecMask(freq_masks=1, freq_width=4,
                                   time_masks=1, time_width=8)},
        ]
    variants = [  # RNN geliştirmeleri
        {'name': 'feature_noise', 'model_cfg': model_cfg,  # öznitelik gürültüsü
         'transform': FeatureNoise(std=0.1)},
    ]
    # Kazanan zaten dikkat havuzlamalı değilse dikkatli varyantları da dene.
    if model_cfg.pooling != 'attn':  # havuzlama zaten dikkat değilse
        attn_cfg = replace(model_cfg, pooling='attn')  # dikkat havuzlamalı sürüm
        variants.insert(0, {'name': 'attention_pooling', 'model_cfg': attn_cfg,  # yalnız dikkat
                            'transform': None})
        variants.append({'name': 'attention_plus_noise', 'model_cfg': attn_cfg,  # dikkat + gürültü
                         'transform': FeatureNoise(std=0.1)})
    return variants  # varyant listesi


def improve_method(
    method: str,  # 'cnn' ya da 'rnn'
    folds: dict[str, pd.DataFrame],  # katmanlar
    *,
    method_dir: Path,  # yöntemin çıktı klasörü
    cache_root: Path,  # önbellek kökü
    device: torch.device,  # cpu/cuda
    max_epochs: int,  # en fazla epoch
    feature_workers: int,  # öznitelik işçisi
    loader_workers: int,  # veri işçisi
    amp: bool,  # karışık hassasiyet
    seed: int,  # tohum
) -> dict[str, Any]:  # Tek yöntemin geliştirme aşamasını koşturur ve özetini döndürür.

    # 1) Kazananı diskteki winner.json'dan oku.
    winner_path = method_dir / 'winner.json'  # kazanan ayar dosyası
    if not winner_path.is_file():  # yoksa
        raise FileNotFoundError(
            f'{winner_path} yok; önce final/run_experiment.py çalıştırın.'  # hata
        )
    with open(winner_path, encoding='utf-8') as handle:  # kazananı oku
        winner = json.load(handle)

    if method == 'cnn':  # CNN ise
        feature_cfg = MelImageConfig(**winner['feature_config'])  # mel ayarı
        extract_fn = extract_mel_image  # mel çıkarıcı
        feature_axis = 1  # normalizasyon ekseni
    else:  # RNN ise
        feature_cfg = IntervalConfig(**winner['feature_config'])  # aralık ayarı
        extract_fn = extract_interval_series  # aralık çıkarıcı
        feature_axis = 2  # normalizasyon ekseni
    base_model_cfg = _model_config_from_dict(method, winner['model_config'])  # kazanan model ayarı
    base_val_f1 = float(winner['val_metrics']['macro_f1'])   # geçilecek çıta

    # 2) Kazananın öznitelikleri (önbellekten anında gelir).
    feature_cache: dict = {}  # koşu-içi bellek
    tensors = _feature_folds(  # katman tensörleri
        folds, feature_cfg, extract_fn, cache_root,
        workers=feature_workers, cache=feature_cache,
    )
    train_x, train_y = tensors['train']  # eğitim
    val_x, val_y = tensors['val']  # geçerleme
    test_x, test_y = tensors['test']  # test
    standardizer = Standardizer.fit(train_x, feature_axis)  # eğitimden normalizasyon
    train_std = standardizer.transform(train_x)  # eğitimi normalize et
    val_std = standardizer.transform(val_x)  # geçerlemeyi normalize et

    def build_model(model_cfg):  # ayardan taze model kurar
        if method == 'cnn':  # CNN ise
            return MelCNN(NUM_CLASSES, model_cfg)  # mel-CNN
        return SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)  # BiGRU

    # 3) Her varyantı eğit ve geçerlemede tabanla kıyasla.
    rows: list[dict[str, Any]] = []  # varyant sonuç satırları
    best: dict[str, Any] | None = None  # geçerlemede en iyi varyant
    for variant in _variants(method, base_model_cfg):  # her varyant için
        started = time.perf_counter()  # süre başlat
        outcome = train_with_early_stopping(  # varyantı eğit
            build_model(variant['model_cfg']),
            train_std,
            train_y,
            val_std,
            val_y,
            variant['model_cfg'].optim,
            num_classes=NUM_CLASSES,
            device=device,
            max_epochs=max_epochs,
            seed=seed,
            num_workers=loader_workers,
            amp=amp,
            train_transform=variant['transform'],   # artırma SADECE eğitimde
        )
        seconds = time.perf_counter() - started  # geçen süre
        row = {  # varyant sonuç satırı
            'variant': variant['name'],  # varyant adı
            'val_accuracy': outcome.validation_metrics['accuracy'],  # geçerleme doğruluğu
            'val_balanced_accuracy': outcome.validation_metrics['balanced_accuracy'],  # dengeli
            'val_macro_f1': outcome.validation_metrics['macro_f1'],  # macro-F1
            'val_weighted_f1': outcome.validation_metrics['weighted_f1'],  # ağırlıklı F1
            'delta_vs_winner': outcome.validation_metrics['macro_f1'] - base_val_f1,  # tabana göre fark
            'best_epoch': outcome.best_epoch,  # en iyi epoch
            'epochs_trained': outcome.epochs_trained,  # koşulan epoch
            'seconds': round(seconds, 1),  # süre
        }
        rows.append(row)  # satırı ekle
        log.info('[%s/improve %s] val macro-F1=%.4f (taban %.4f, fark %+.4f)',  # logla
                 method, variant['name'], row['val_macro_f1'], base_val_f1,
                 row['delta_vs_winner'])
        if best is None or row['val_macro_f1'] > best['row']['val_macro_f1']:  # bu varyant daha iyiyse
            best = {'row': row, 'variant': variant, 'outcome': outcome}  # en iyiyi güncelle

    # 4) Varyant tablosunu diske yaz.
    improvements = pd.DataFrame(rows)  # satırları tabloya çevir
    improvements.insert(0, 'method', method)  # yöntem sütunu ekle
    improvements.to_csv(method_dir / 'improvements.csv', index=False)  # CSV kaydet

    summary: dict[str, Any] = {  # geliştirme özeti
        'method': method,
        'baseline_val_macro_f1': base_val_f1,  # taban çıta
        'variants': rows,  # tüm varyant sonuçları
        'improved_on_validation': bool(best and best['row']['delta_vs_winner'] > 0),  # tabanı geçen var mı
    }
    # 5) Test disiplini: yalnızca geçerlemede tabanı GEÇEN varyant teste gider.
    if best and best['row']['delta_vs_winner'] > 0:  # geçerlemede iyileşme varsa
        class_weights = inverse_frequency_weights(  # sınıf ağırlıkları
            folds['train']['label_idx'].to_numpy(), NUM_CLASSES
        )
        _, _, test_prob = evaluate_arrays(  # en iyi varyantı test'te değerlendir
            best['outcome'].model,
            standardizer.transform(test_x),
            test_y,
            class_weights=class_weights,
            device=device,
            num_workers=loader_workers,
        )
        test_metrics = evaluate_report(  # test metrikleri + matris yaz
            test_y, test_prob.argmax(axis=1), method_dir, prefix='test_improved',
            title=f'{method.upper()} improved test confusion matrix',
        )
        torch.save(  # geliştirilmiş modeli kaydet
            {
                'state_dict': best['outcome'].model.state_dict(),
                'variant': best['variant']['name'],
                'feature_config': feature_cfg.__dict__,
                'model_config': best['variant']['model_cfg'].to_dict(),
                'standardizer_mean': standardizer.mean,
                'standardizer_scale': standardizer.scale,
                'feature_axis': standardizer.feature_axis,
            },
            method_dir / 'improved_model.pt',
        )
        summary['best_variant'] = best['variant']['name']  # kazanan varyant adı
        summary['test_improved'] = test_metrics  # test sonucu
        log.info('[%s] improved TEST acc=%.4f macro-F1=%.4f',  # logla
                 method, test_metrics['accuracy'], test_metrics['macro_f1'])
    else:  # hiçbir varyant tabanı geçemediyse
        log.info('[%s] no variant beat the winner on validation; '  # test'e dokunma
                 'test left untouched.', method)

    with open(method_dir / 'improvements_summary.json', 'w', encoding='utf-8') as handle:  # özeti yaz
        json.dump(summary, handle, indent=2)
    return summary  # özeti döndür


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--manifest', default='data/processed/manifest.csv')  # manifest
    parser.add_argument('--cache-root', default='data/cache/final')  # önbellek
    parser.add_argument('--out-root', default='final/outputs')  # çıktı kökü
    parser.add_argument('--corpus', default='cremad', choices=['cremad', 'meld'])  # veri seti
    parser.add_argument(  # yöntemler
        '--methods', nargs='+', default=['cnn', 'rnn'], choices=['cnn', 'rnn']
    )
    parser.add_argument('--max-epochs', type=int, default=60)  # en fazla epoch
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')  # cihaz
    parser.add_argument('--feature-workers', type=int, default=4)  # öznitelik işçisi
    parser.add_argument('--loader-workers', type=int, default=0)  # veri işçisi
    parser.add_argument('--seed', type=int, default=42)  # tohum
    parser.add_argument('--no-amp', action='store_true')  # AMP kapat
    parser.add_argument(  # tanı amaçlı küçültme
        '--limit-per-split',
        type=int,
        help='YALNIZ TANI: her katmanı oransal olarak bu satır sayısına indir.',
    )
    args = parser.parse_args()  # argümanları oku

    # Bölme, ana deneyle AYNI ayarlarla (aynı seed) yeniden kurulur;
    # deterministik olduğu için katmanlar birebir aynı çıkar.
    manifest = pd.read_csv(args.manifest)  # manifest
    settings = SplitSettings(train_corpora=(args.corpus,), eval_corpora=(args.corpus,))  # bölme ayarı
    train_df, val_df, test_df = prepare_splits(manifest, settings, seed=args.seed)  # aynı bölme
    if args.limit_per_split is not None:  # tanı küçültmesi istendiyse
        train_df = _limit_stratified(train_df, args.limit_per_split, args.seed)  # eğitimi küçült
        val_df = _limit_stratified(val_df, args.limit_per_split, args.seed + 1)  # geçerlemeyi küçült
        test_df = _limit_stratified(test_df, args.limit_per_split, args.seed + 2)  # testi küçült
        log.warning('Diagnostic limit active: train=%d val=%d test=%d',  # uyar
                    len(train_df), len(val_df), len(test_df))
    folds = {'train': train_df, 'val': val_df, 'test': test_df}  # katmanlar

    if args.device == 'auto':  # otomatik cihaz
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    else:  # elle verildiyse
        device = torch.device(args.device)  # o cihaz

    corpus_dir = ensure_dir(Path(args.out_root) / args.corpus)  # çıktı/corpus klasörü
    for method in args.methods:  # her yöntem için
        improve_method(  # geliştirme aşamasını koştur
            method,
            folds,
            method_dir=Path(corpus_dir) / method,
            cache_root=Path(args.cache_root),
            device=device,
            max_epochs=args.max_epochs,
            feature_workers=args.feature_workers,
            loader_workers=args.loader_workers,
            amp=not args.no_amp,
            seed=args.seed,
        )


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
