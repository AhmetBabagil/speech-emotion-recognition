# Tek corpus için uçtan uca final deney hattı.
#
# Akış (her yöntem için ayrı ayrı):
# 1. Manifest'i oku, konuşmacı-bağımsız eğitim/geçerleme/test bölmesi yap.
# 2. Adayların özniteliklerini çıkar (önbellekten geliyorsa saniyeler sürer).
# 3. ARAMA: her adayı eğit, geçerleme macro-F1'ine göre sırala.
# 4. İYİLEŞTİRME TURU: kazananın çevresindeki yerel adayları da dene.
# 5. Nihai kazananı TEST kümesinde BİR KEZ değerlendir.
# 6. Tüm artefaktları (arama logu, kazanan ayarlar, öğrenme eğrisi, metrikler, karışıklık matrisi, model ağırlıkları) çıktı klasörüne yaz.
#
# Dürüstlük ilkesi: test kümesine yalnızca seçilmiş nihai model dokunur; hiperparametre kararlarının hiçbiri test sonucuna bakılarak verilmez.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from dataclasses import dataclass  # SplitSettings'i kolay yazmak için
import json  # ayar ve sonuçları JSON'a yazmak için
from pathlib import Path  # dosya yolları
import time  # eğitim süresini ölçmek için
from typing import Any, Callable  # tip ipuçları

import numpy as np  # sayısal diziler
import pandas as pd  # tablo (manifest, arama logu)
import torch  # cihaz seçimi, model kaydı

from final.dataset import Standardizer, load_feature_tensor  # normalizasyon + öznitelik yükleme
from final.features import (  # öznitelik ayarları + çıkarıcılar
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import MelCNN, RNNConfig, SeqRNN, count_parameters  # modeller + parametre sayacı
from final.search_space import (  # hiperparametre aday listeleri
    cnn_refinement,
    cnn_space,
    rnn_refinement,
    rnn_space,
)
from final.training import (  # eğitim + değerlendirme
    TrainingOutcome,
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.data.splits import prepare_splits  # konuşmacı-bağımsız bölme
from ser.constants import NUM_CLASSES  # sınıf sayısı (6)
from ser.evaluate import report as evaluate_report  # metrik + karışıklık matrisi yazan yardımcı
from ser.utils import ensure_dir, get_logger  # klasör oluşturma + günlükleme

log = get_logger(__name__)  # bu modülün günlükleyicisi


@dataclass(frozen=True)  # kilitli ayar sınıfı
class SplitSettings:
    # ser.config'in veri bölümünün, prepare_splits'e yetecek kadarı.
    #
    # split="speaker": bölme aktör kimliğine göre yapılır — bir konuşmacının tüm kayıtları tek katmanda kalır (konuşmacı-bağımsız protokol).

    train_corpora: tuple[str, ...]  # eğitim veri setleri
    eval_corpora: tuple[str, ...]  # değerlendirme veri setleri
    val_fraction: float = 0.15  # geçerleme oranı
    test_fraction: float = 0.15  # test oranı
    split: str = 'speaker'  # bölme türü: konuşmacıya göre


def _limit_stratified(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    # SADECE TANI amaçlı: katmanı, sınıf oranlarını koruyarak küçültür.
    #
    # Duman testlerinde (quick mod) tüm hattı dakikalar içinde uçtan uca denemek için kullanılır; gerçek deneylerde devrede değildir.

    if limit >= len(df):  # zaten küçükse
        return df  # dokunma
    parts = []  # sınıf başına küçültülmüş parçalar
    rng_seed = seed  # sınıflar için değişen tohum
    for _, group in df.groupby('label_idx'):  # her sınıf için
        # Her sınıftan, o sınıfın genel orandaki payı kadar örnek al.
        take = max(1, int(round(limit * len(group) / len(df))))  # oransal örnek sayısı
        parts.append(group.sample(n=min(take, len(group)), random_state=rng_seed))  # o kadar örnek çek
        rng_seed += 1  # tohumu değiştir
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)  # birleştir + karıştır


def _feature_folds(
    folds: dict[str, pd.DataFrame],  # katman adı -> kayıtlar
    feature_cfg,  # öznitelik ayarı
    extract_fn: Callable,  # öznitelik çıkaran fonksiyon
    cache_root: str | Path,  # önbellek kökü
    *,
    workers: int,  # paralel işçi
    cache: dict,  # koşu-içi bellek
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Verilen katmanların öznitelik tensörlerini üretir (veya yeniden kullanır).
    #
    # ``cache`` sözlüğü koşu-içi bellektir: aynı öznitelik ayarını kullanan birden çok aday, katman tensörlerini yalnızca bir kez hesaplatır.

    result = {}  # katman adı -> (öznitelik, etiket)
    for name, records in folds.items():  # her katman için
        key = (feature_cfg.fingerprint, name)  # bellek anahtarı: ayar kimliği + katman
        if key not in cache:  # daha önce hesaplanmadıysa
            cache[key] = load_feature_tensor(  # öznitelikleri yükle/çıkar
                records,  # katmanın kayıtları
                cache_root,  # önbellek kökü
                feature_cfg.fingerprint,  # ayar kimliği
                lambda path: extract_fn(path, feature_cfg),  # tek dosya çıkarıcı
                feature_cfg.shape,  # beklenen boyut
                workers=workers,  # paralel işçi
                description=f'{feature_cfg.fingerprint} {name}',  # ilerleme açıklaması
            )
        result[name] = cache[key]  # sonucu al (bellekten ya da yeni)
    return result  # katman tensörleri


def _candidate_row(feature_cfg, model_cfg, outcome: TrainingOutcome, seconds: float,
                   stage: str, n_params: int) -> dict[str, Any]:  # Bir adayın tüm sonuçlarını arama loguna yazılacak tek satıra çevirir.

    row = {  # aday sonucunun tek satırlık kaydı
        'stage': stage,                                              # 'search' / 'refine'
        'feature_fingerprint': feature_cfg.fingerprint,  # öznitelik ayar kimliği
        'feature_config': json.dumps(feature_cfg.__dict__, sort_keys=True),  # öznitelik ayarı (JSON)
        'model_config': json.dumps(model_cfg.to_dict(), sort_keys=True),  # model ayarı (JSON)
        'parameters': n_params,  # model parametre sayısı
        'best_epoch': outcome.best_epoch,  # en iyi epoch
        'epochs_trained': outcome.epochs_trained,  # koşulan epoch
        'stopped_early': outcome.stopped_early,  # erken mi durdu
        'val_loss': outcome.validation_loss,  # geçerleme kaybı
        'val_accuracy': outcome.validation_metrics['accuracy'],  # geçerleme doğruluğu
        'val_balanced_accuracy': outcome.validation_metrics['balanced_accuracy'],  # dengeli doğruluk
        'val_macro_f1': outcome.validation_metrics['macro_f1'],  # geçerleme macro-F1 (sıralama ölçütü)
        'val_weighted_f1': outcome.validation_metrics['weighted_f1'],  # ağırlıklı F1
        'seconds': round(seconds, 1),  # eğitim süresi
    }
    return row  # log satırı


def _plot_history(history: list[dict[str, Any]], out_path: Path, title: str) -> None:
    # Kazanan adayın öğrenme eğrilerini (loss + macro-F1) PNG olarak kaydeder.
    #
    # Bu grafik, erken durdurmanın çalıştığının görsel kanıtıdır: eğitim kaybı düşmeye devam ederken geçerleme kaybının dönmesi = aşırı öğrenme.

    import matplotlib  # grafik kütüphanesi
    matplotlib.use('Agg')   # ekran gerektirmeyen arka uç (sunucu/terminal uyumlu)
    import matplotlib.pyplot as plt  # çizim arayüzü

    epochs = [h['epoch'] for h in history]  # x ekseni: epoch numaraları
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # yan yana iki grafik
    ax1.plot(epochs, [h['train_loss'] for h in history], label='train')  # eğitim kaybı eğrisi
    ax1.plot(epochs, [h['val_loss'] for h in history], label='validation')  # geçerleme kaybı eğrisi
    ax1.set_xlabel('Epoch')  # x etiketi
    ax1.set_ylabel('Weighted CE loss')  # y etiketi
    ax1.legend()  # açıklama kutusu
    ax1.grid(alpha=0.3)  # hafif ızgara
    ax2.plot(epochs, [h['train_macro_f1'] for h in history], label='train')  # eğitim macro-F1
    ax2.plot(epochs, [h['val_macro_f1'] for h in history], label='validation')  # geçerleme macro-F1
    ax2.set_xlabel('Epoch')  # x etiketi
    ax2.set_ylabel('Macro F1')  # y etiketi
    ax2.legend()  # açıklama
    ax2.grid(alpha=0.3)  # ızgara
    fig.suptitle(title)  # üst başlık
    fig.tight_layout()  # yerleşimi sıkılaştır
    fig.savefig(out_path, dpi=150)  # PNG olarak kaydet
    plt.close(fig)  # belleği boşalt


def run_method(
    method: str,  # 'cnn' ya da 'rnn'
    folds: dict[str, pd.DataFrame],  # eğitim/geçerleme/test kayıtları
    *,
    cache_root: Path,  # önbellek kökü
    out_dir: Path,  # çıktı klasörü
    grid_mode: str,  # arama modu (quick/report)
    max_epochs: int,  # en fazla epoch
    device: torch.device,  # cpu/cuda
    feature_workers: int,  # öznitelik işçisi
    loader_workers: int,  # veri işçisi
    amp: bool,  # karışık hassasiyet
    refine: bool,  # iyileştirme turu yapılsın mı
    seed: int,  # tohum
) -> dict[str, Any]:  # Tek bir yöntemi arar, iyileştirir ve test eder; özet sözlüğü döndürür.

    # Yönteme göre aday listesi, öznitelik fonksiyonu ve normalizasyon ekseni seç.
    if method == 'cnn':  # Yöntem 1
        candidates = cnn_space(grid_mode)  # CNN aday listesi
        extract_fn = extract_mel_image  # mel görüntüsü çıkarıcı
        feature_axis = 1  # [N, mels, T] -> mel bandı başına istatistik
        refinement_fn = cnn_refinement  # CNN iyileştirme adayları
    elif method == 'rnn':  # Yöntem 2
        candidates = rnn_space(grid_mode)  # RNN aday listesi
        extract_fn = extract_interval_series  # aralık serisi çıkarıcı
        feature_axis = 2  # [N, T, D] -> öznitelik boyutu başına istatistik
        refinement_fn = rnn_refinement  # RNN iyileştirme adayları
    else:  # tanımsız yöntem
        raise ValueError(f'Bilinmeyen yöntem {method!r}.')  # hata

    def build_model(feature_cfg, model_cfg):  # Adayın ayarlarından taze (rastgele başlatılmış) bir model kurar.

        if method == 'cnn':  # CNN ise
            return MelCNN(NUM_CLASSES, model_cfg)  # mel-CNN kur
        return SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)  # değilse BiGRU kur

    ensure_dir(out_dir)  # çıktı klasörünü oluştur
    feature_cache: dict = {}                 # koşu-içi öznitelik belleği
    rows: list[dict[str, Any]] = []          # arama logunun satırları
    best: dict[str, Any] | None = None       # şu ana kadarki geçerleme kazananı

    def run_stage(stage: str, stage_candidates) -> None:  # Bir aday listesini eğitir; geçerleme kazananını `best`te günceller.

        nonlocal best  # dış fonksiyondaki best'i güncelle
        for index, (feature_cfg, model_cfg) in enumerate(stage_candidates, start=1):  # her aday için
            # 1) Bu adayın öznitelikleri (eğitim + geçerleme katmanları).
            tensors = _feature_folds(
                {'train': folds['train'], 'val': folds['val']},  # sadece eğitim + geçerleme
                feature_cfg,
                extract_fn,
                cache_root,
                workers=feature_workers,
                cache=feature_cache,
            )
            train_x, train_y = tensors['train']  # eğitim öznitelik/etiket
            val_x, val_y = tensors['val']  # geçerleme öznitelik/etiket
            # 2) Normalizasyon YALNIZ eğitim istatistikleriyle öğrenilir.
            standardizer = Standardizer.fit(train_x, feature_axis)  # sızıntısız normalizasyon
            model = build_model(feature_cfg, model_cfg)  # taze model kur
            n_params = count_parameters(model)  # parametre sayısını al
            started = time.perf_counter()  # süre ölçümü başlat
            # 3) Ağırlıklı loss + early stopping ile eğit.
            outcome = train_with_early_stopping(
                model,
                standardizer.transform(train_x),  # normalize edilmiş eğitim
                train_y,
                standardizer.transform(val_x),  # normalize edilmiş geçerleme
                val_y,
                model_cfg.optim,
                num_classes=NUM_CLASSES,
                device=device,
                max_epochs=max_epochs,
                seed=seed,
                num_workers=loader_workers,
                amp=amp,
            )
            seconds = time.perf_counter() - started  # geçen süre
            # 4) Sonucu logla ve gerekirse kazananı güncelle.
            row = _candidate_row(feature_cfg, model_cfg, outcome, seconds, stage, n_params)  # log satırı
            rows.append(row)  # loga ekle
            log.info(  # ilerlemeyi ekrana yaz
                '[%s/%s %d/%d] val macro-F1=%.4f acc=%.4f (epoch %d/%d, %.0fs)',
                method, stage, index, len(stage_candidates),
                row['val_macro_f1'], row['val_accuracy'],
                outcome.best_epoch, outcome.epochs_trained, seconds,
            )
            if best is None or row['val_macro_f1'] > best['row']['val_macro_f1']:  # bu aday daha iyiyse
                best = {  # kazananı güncelle
                    'row': row,
                    'feature_cfg': feature_cfg,
                    'model_cfg': model_cfg,
                    'standardizer': standardizer,
                    'outcome': outcome,
                }

    # ---- Aşama 1: geniş arama ----
    run_stage('search', candidates)  # tüm adayları dene
    # ---- Aşama 2: kazananın çevresinde yerel iyileştirme ----
    if refine and grid_mode != 'quick' and best is not None:  # iyileştirme açıksa
        # Daha önce denenen adaylar tekrar eğitilmesin.
        seen = set(candidates) | {(best['feature_cfg'], best['model_cfg'])}  # denenmiş adaylar
        refinement = [  # kazananın çevresindeki yeni adaylar
            c for c in refinement_fn((best['feature_cfg'], best['model_cfg']))
            if c not in seen  # daha önce denenmemiş olanlar
        ]
        if refinement:  # yeni aday varsa
            run_stage('refine', refinement)  # iyileştirme turunu koştur

    if best is None:  # hiç aday başarısız olduysa
        raise RuntimeError(f'{method} için başarılı aday yok.')  # hata

    # Arama logunu diske yaz (rapor/sunum tabloları buradan üretilir).
    search_log = pd.DataFrame(rows)  # log satırlarını tabloya çevir
    search_log.to_csv(out_dir / 'search_log.csv', index=False)  # CSV olarak kaydet

    feature_cfg = best['feature_cfg']  # kazananın öznitelik ayarı
    model_cfg = best['model_cfg']  # kazananın model ayarı
    standardizer = best['standardizer']  # kazananın normalizasyonu
    outcome: TrainingOutcome = best['outcome']  # kazananın eğitim sonucu

    # Kazananın öğrenme geçmişi + eğrisi + ayarları.
    pd.DataFrame(outcome.history).to_csv(out_dir / 'winner_history.csv', index=False)  # eğitim geçmişi CSV
    _plot_history(outcome.history, out_dir / 'winner_learning_curve.png',  # öğrenme eğrisi PNG
                  f'{method.upper()} winner learning curve')
    with open(out_dir / 'winner.json', 'w', encoding='utf-8') as handle:  # kazanan ayarları JSON
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
    torch.save(  # kazanan modeli diske kaydet
        {
            'state_dict': outcome.model.state_dict(),  # ağırlıklar
            'feature_config': feature_cfg.__dict__,  # öznitelik ayarı
            'model_config': model_cfg.to_dict(),  # model ayarı
            'standardizer_mean': standardizer.mean,  # normalizasyon ortalaması
            'standardizer_scale': standardizer.scale,  # normalizasyon std'si
            'feature_axis': standardizer.feature_axis,  # normalizasyon ekseni
        },
        out_dir / 'winner_model.pt',
    )

    # TEST katmanına tam olarak bir kez, yalnızca nihai kazanan dokunur.
    test_x, test_y = _feature_folds(  # test özniteliklerini çıkar
        {'test': folds['test']},
        feature_cfg,
        extract_fn,
        cache_root,
        workers=feature_workers,
        cache=feature_cache,
    )['test']
    class_weights = inverse_frequency_weights(  # sınıf ağırlıkları (eğitimden)
        folds['train']['label_idx'].to_numpy(), NUM_CLASSES
    )
    test_loss, _, test_prob = evaluate_arrays(  # modeli test'te değerlendir
        outcome.model,
        standardizer.transform(test_x),  # aynı normalizasyonla
        test_y,
        class_weights=class_weights,
        device=device,
        num_workers=loader_workers,
    )
    test_pred = test_prob.argmax(axis=1)  # olasılıklardan tahminler
    # evaluate_report: metrics.json + karışıklık matrisi PNG'sini yazar.
    test_metrics = evaluate_report(  # test metriklerini + matrisi kaydet
        test_y, test_pred, out_dir, prefix='test',
        title=f'{method.upper()} test confusion matrix',
    )
    val_metrics = outcome.validation_metrics  # geçerleme metrikleri
    log.info('[%s] TEST acc=%.4f macro-F1=%.4f (val macro-F1=%.4f)',  # sonucu logla
             method, test_metrics['accuracy'], test_metrics['macro_f1'],
             val_metrics['macro_f1'])
    return {  # yöntemin özet sonucu
        'method': method,
        'winner_feature': feature_cfg.__dict__,
        'winner_model': model_cfg.to_dict(),
        'val': val_metrics,
        'test': test_metrics,
        'test_loss': test_loss,
        'search_rows': len(rows),
    }


def run_all(
    manifest_path: str | Path,  # manifest CSV yolu
    cache_root: str | Path,  # önbellek kökü
    output_root: str | Path,  # çıktı kökü
    *,
    corpus: str = 'cremad',  # veri seti
    methods: tuple[str, ...] = ('cnn', 'rnn'),  # çalıştırılacak yöntemler
    grid_mode: str = 'report',  # arama modu
    max_epochs: int = 60,  # en fazla epoch
    device_name: str = 'auto',  # cihaz
    feature_workers: int = 1,  # öznitelik işçisi
    loader_workers: int = 0,  # veri işçisi
    amp: bool = True,  # karışık hassasiyet
    refine: bool = True,  # iyileştirme turu
    limit_per_split: int | None = None,  # duman testi limiti
    prior_results_path: str | Path | None = None,  # önceki sonuçlar
    seed: int = 42,  # tohum
) -> dict[str, dict[str, Any]]:  # Tüm deneyi koşturur: bölme -> her yöntem -> karşılaştırma tablosu.

    # 1) Manifest + konuşmacı-bağımsız bölme (deterministik: seed=42).
    manifest = pd.read_csv(manifest_path)  # manifesti oku
    settings = SplitSettings(train_corpora=(corpus,), eval_corpora=(corpus,))  # bölme ayarı
    train_df, val_df, test_df = prepare_splits(manifest, settings, seed=seed)  # konuşmacı-bağımsız böl
    if limit_per_split is not None:  # duman testi isteniyorsa
        # Yalnız duman testi: katmanları oransal küçült.
        train_df = _limit_stratified(train_df, limit_per_split, seed)  # eğitimi küçült
        val_df = _limit_stratified(val_df, limit_per_split, seed + 1)  # geçerlemeyi küçült
        test_df = _limit_stratified(test_df, limit_per_split, seed + 2)  # testi küçült
        log.warning('Diagnostic limit active: train=%d val=%d test=%d',  # uyarı
                    len(train_df), len(val_df), len(test_df))
    folds = {'train': train_df, 'val': val_df, 'test': test_df}  # katmanlar sözlüğü

    # 2) Cihaz seçimi: varsa GPU, yoksa CPU.
    if device_name == 'auto':  # otomatik seçim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU varsa GPU
    else:  # elle verildiyse
        device = torch.device(device_name)  # o cihazı kullan
    log.info('Corpus=%s device=%s grid=%s | train=%d val=%d test=%d',  # özet logla
             corpus, device, grid_mode, len(train_df), len(val_df), len(test_df))

    # 3) Her yöntemi sırayla koştur; çıktılar corpus/yöntem klasörlerine gider.
    output_root = ensure_dir(Path(output_root) / corpus)  # çıktı/corpus klasörü
    cache_root = Path(cache_root)  # önbellek kökü
    results: dict[str, dict[str, Any]] = {}  # yöntem -> sonuç
    for method in methods:  # her yöntem için
        results[method] = run_method(  # yöntemi koştur
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
    comparison = _comparison_table(results, prior_results_path)  # karşılaştırma tablosu
    comparison.to_csv(Path(output_root) / 'method_comparison.csv', index=False)  # CSV kaydet
    with open(Path(output_root) / 'summary.json', 'w', encoding='utf-8') as handle:  # özet JSON
        json.dump(results, handle, indent=2, default=str)  # sonuçları yaz
    return results  # tüm sonuçlar


def _comparison_table(
    results: dict[str, dict[str, Any]],  # yöntem sonuçları
    prior_results_path: str | Path | None,  # eski sonuç CSV'si (opsiyonel)
) -> pd.DataFrame:  # Yöntem karşılaştırma CSV'sini kurar; istenirse eski sonuçları da ekler.

    rows = []  # tablo satırları
    names = {'cnn': 'Yöntem 1: Mel + CNN', 'rnn': 'Yöntem 2: Aralık + LSTM/GRU'}  # okunur adlar
    for method, result in results.items():  # her yöntem için
        rows.append({  # bir karşılaştırma satırı
            'model': names.get(method, method),  # okunur ad
            'source': 'final',  # kaynak etiketi
            'val_macro_f1': result['val']['macro_f1'],  # geçerleme macro-F1
            'test_accuracy': result['test']['accuracy'],  # test doğruluğu
            'test_balanced_accuracy': result['test']['balanced_accuracy'],  # dengeli doğruluk
            'test_macro_f1': result['test']['macro_f1'],  # test macro-F1
            'test_weighted_f1': result['test']['weighted_f1'],  # ağırlıklı F1
        })
    table = pd.DataFrame(rows)  # satırları tabloya çevir
    if prior_results_path and Path(prior_results_path).is_file():  # eski sonuç verildiyse
        try:
            prior = pd.read_csv(prior_results_path)  # eski tabloyu oku
            prior['source'] = Path(prior_results_path).parent.name  # kaynağını etiketle
            table = pd.concat([table, prior], ignore_index=True)  # iki tabloyu birleştir
        except (OSError, ValueError, pd.errors.ParserError) as error:  # okuma hatası olursa
            log.warning('Önceki sonuçlar birleştirilemedi: %s', error)  # uyar, devam et
    return table  # karşılaştırma tablosu
