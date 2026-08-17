"""Ödev 2 klasik makine öğrenmesi boru hattı (Decision Tree / Random Forest / Gradient Boosting).

Her veri seti (corpus) ve her model için izlenen akış:
  - Wav2Vec2 havuzlanmış öznitelik boyutu (F) seçilir,
  - arama PCA'lı ve PCA'sız olarak tekrarlanır (P),
  - ödevde istenen model hiperparametreleri geçerleme (validation) verisinde optimize edilir,
  - en iyi yapılandırma train+validation üzerinde yeniden fit edilir,
  - test verisinde yalnızca BİR kez değerlendirme yapılır.

Bu protokol Ödev 1'deki KNN çalışmasıyla birebir aynıdır; amaç, model aileleri
arasında adil bir karşılaştırma yapabilmektir (aynı bölmeler, aynı öznitelikler,
aynı metrikler).

Makine öğrenmesi iş akışı yalnızca numpy, pandas ve scikit-learn kullanır.
Wav2Vec2 burada ÇALIŞTIRILMAZ; bu dosya sadece Ödev 1'de üretilip cache'lenen
vektörleri okur. Rapor modu çıktıları, teslim incelemesi için odev2/outputs
altında tutulur.
"""

from __future__ import annotations

import os

# BLAS/OpenMP iş parçacığı sayıları, DAHA sklearn/numpy import edilmeden 1'e
# sabitlenir (bu yüzden bu satırlar dosyanın en üstündedir). Nedeni: Windows'ta
# ve kısıtlı ortamlarda bu kütüphanelerin çok iş parçacıklı çalışması hem CPU'yu
# aşırı doldurabiliyor hem de izin/kararlılık sorunları çıkarabiliyor.
# setdefault kullanıldığı için, kullanıcı bu değişkenleri dışarıdan ayarlamışsa
# ona dokunulmaz — yalnızca boşsa 1 yazılır.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Proje kökünü arama yoluna ekle: `ser` ve `odev1` paketleri doğrudan
# çalıştırmada da bulunabilsin.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ödev 1 ile ortak parçalar bilerek yeniden kullanılır: metrikler, çizim,
# öznitelik yükleme ve bölme mantığı iki ödevde birebir aynı olmalıdır.
from odev1.evaluation import compute_metrics, plot_confusion  # noqa: E402
from odev1.features_w2v import load_pooled  # noqa: E402
from ser.config import Config  # noqa: E402
from ser.constants import CORPUS_CREMAD, CORPUS_MELD  # noqa: E402
from ser.data import prepare_splits  # noqa: E402
from ser.utils import ensure_dir, get_logger, set_seed  # noqa: E402

log = get_logger("odev2.models")

SEED = 42  # tüm rastgele işlemler bu tohumu kullanır → deneyler tekrarlanabilir

# ---- Ortak arama uzayları ---------------------------------------------------
# F: Ödev 1 ile aynı üç havuzlama seçeneği ve boyutları.
POOLS = ["mean", "mean_std", "mean_std_max"]
POOL_DIM = {"mean": 768, "mean_std": 1536, "mean_std_max": 2304}
# P: 0 "PCA yok" demektir; kalanlar hedef bileşen sayılarıdır.
PCA_DIMS = [0, 32, 64, 128, 256, 512]

# Üç ızgara modu: quick küçük bir duman testi, report teslim için dengeli bir
# arama, full ise en kapsamlı (ve en yavaş) taramadır.
QUICK_POOLS = ["mean"]
QUICK_PCA_DIMS = [0, 64]
REPORT_POOLS = POOLS
REPORT_PCA_DIMS = [0, 64, 128, 256]
GRID_MODES = ("quick", "report", "full")


@dataclass(frozen=True)
class ModelSpec:
    """Bir model ailesinin adını, ızgaralarını ve estimator fabrikasını birlikte tutar.

    Bu veri sınıfı (dataclass) sayesinde Decision Tree, Random Forest ve Gradient
    Boosting aynı deney döngüsünden geçirilebilir; modele özel farklar tek yapıda
    tanımlanır. `frozen=True` nesneyi değiştirilemez yapar — tanımların deney
    sırasında yanlışlıkla bozulmasını engeller.

    Alanların anlamları:
      * ``name``            : dosya adlarında ve CLI'da kullanılan makine adı,
      * ``display_name``    : tablolarda/grafiklerde görünen okunur ad,
      * ``param_grid``      : full mod için hiperparametre listeleri,
      * ``report_grid``     : report modunun daraltılmış listeleri,
      * ``quick_grid``      : hızlı kontrol için en küçük listeler,
      * ``make_estimator``  : parametre sözlüğünden sklearn modeli kuran fonksiyon,
      * ``fit_with_sample_weight`` : sınıf dengesizliği örnek ağırlığıyla mı çözülecek
        (class_weight parametresi olmayan HistGradientBoosting için gerekli).
    """
    name: str
    display_name: str
    param_grid: dict[str, list[Any]]
    report_grid: dict[str, list[Any]]
    quick_grid: dict[str, list[Any]]
    make_estimator: Callable[[dict[str, Any]], Any]
    fit_with_sample_weight: bool = False


def _grid_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Hiperparametre listelerinin Kartezyen çarpımını ayar sözlükleri listesine çevirir.

    Örnek: {"a": [1, 2], "b": [x]} → [{"a": 1, "b": x}, {"a": 2, "b": x}].
    `itertools.product` tüm kombinasyonları üretir; her kombinasyon `zip` ile
    anahtar adlarıyla eşleştirilip sözlük yapılır. Grid search'ün temel taşı.
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _grid_settings(spec: ModelSpec, grid_mode: str) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    """Quick/report/full moduna göre feature, PCA ve model ayar listelerini seçer.

    `quick` kodun uçtan uca çalıştığını görmek için, `report` teslim edilecek
    deneyler için, `full` ise en geniş arama içindir. Report modunda pahalı olan
    Gradient Boosting'e özel bir istisna vardır: yalnızca none/64 PCA denenir,
    aksi hâlde toplam süre orantısız uzardı.
    """
    if grid_mode == "quick":
        return QUICK_POOLS, QUICK_PCA_DIMS, _grid_product(spec.quick_grid)
    if grid_mode == "report":
        pca_dims = [0, 64] if spec.name == "gradient_boosting" else REPORT_PCA_DIMS
        return REPORT_POOLS, pca_dims, _grid_product(spec.report_grid)
    if grid_mode == "full":
        return POOLS, PCA_DIMS, _grid_product(spec.param_grid)
    raise ValueError(f"Unknown grid mode: {grid_mode}. Expected one of {GRID_MODES}.")


def _json_default(value: Any) -> str:
    """NumPy skalerlerini JSON'un yazabileceği yerel Python değerlerine çevirir.

    json.dumps, np.int64 / np.float64 gibi türleri doğrudan yazamaz ve hata
    fırlatır; bu fonksiyon `default=` kancası olarak verilir ve `.item()` ile
    saf Python sayısına çevirir. Tanımadığı her şeyi de metne dönüştürür.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _params_to_text(params: dict[str, Any]) -> str:
    """Hiperparametre sözlüğünü CSV/JSON tablolarında sabit sıralı metne çevirir.

    `sort_keys=True` sayesinde aynı ayar her zaman aynı metni üretir; böylece
    tablolarda satırlar güvenle karşılaştırılabilir ve gruplanabilir.
    """
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=_json_default)


# ---- Model fabrikaları: parametre sözlüğü → kurulmuş sklearn modeli ---------

def _make_decision_tree(params: dict[str, Any]) -> DecisionTreeClassifier:
    """Sınıf dengesini ağırlıklandıran, tekrarlanabilir bir Karar Ağacı kurar.

    `class_weight="balanced"`: az örnekli duygu sınıflarının hataları daha
    pahalı sayılır, model azınlık sınıflarını yok sayamaz.
    `random_state=SEED`: eşit kazançlı bölünmelerde seçim hep aynı olur.
    """
    return DecisionTreeClassifier(random_state=SEED, class_weight="balanced", **params)


def _make_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    """Sınıf ağırlıklı Rastgele Orman kurar; Windows güvenliği için tek iş parçacığı kullanır.

    Rastgele Orman = çok sayıda karar ağacının (her biri farklı örnek ve
    öznitelik alt kümesiyle eğitilip) oy birliği; tek ağaca göre varyansı düşürür.
    """
    return RandomForestClassifier(
        random_state=SEED,
        class_weight="balanced",
        # n_jobs=1: kısıtlı Windows ortamlarında paralel süreç başlatma izin
        # hatalarına yol açabildiği için paralellik bilerek kapalı tutulur.
        n_jobs=1,
        **params,
    )


def _make_gradient_boosting(params: dict[str, Any]) -> HistGradientBoostingClassifier:
    """Histogram tabanlı, early-stopping kullanan Gradient Boosting modeli kurar.

    Gradient Boosting, ağaçları sırayla ekler; her yeni ağaç öncekilerin
    hatasını düzeltmeye çalışır. "Hist" sürümü öznitelikleri kutulara (bin)
    ayırarak büyük veride çok daha hızlıdır. Süreyi sınırlamak için:
      * ``max_iter=30``      : en fazla 30 ağaç turu,
      * ``max_bins=32``      : kaba ama hızlı histogramlar,
      * ``early_stopping=True``: iç doğrulama skoru iyileşmeyince erken durur.
    """
    return HistGradientBoostingClassifier(
        random_state=SEED,
        max_iter=30,
        max_bins=32,
        early_stopping=True,
        **params,
    )


# ---- Deneyde yarışan üç model ailesi ve hiperparametre ızgaraları -----------
MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="decision_tree",
        display_name="Karar Agaci",
        # criterion: bölünme kalitesi ölçütü; max_depth: aşırı öğrenmeyi sınırlar
        # (None = sınırsız); min_samples_split: bir düğümün bölünebilmesi için
        # gereken en az örnek sayısı (büyük değer = daha sade ağaç).
        param_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 4, 8, 12, 16, 24],
            "min_samples_split": [2, 5, 10, 20],
        },
        report_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 10],
        },
        quick_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 8],
            "min_samples_split": [2, 10],
        },
        make_estimator=_make_decision_tree,
    ),
    ModelSpec(
        name="random_forest",
        display_name="Rastgele Orman",
        # n_estimators: ormandaki ağaç sayısı; max_features: her bölünmede
        # rastgele değerlendirilecek öznitelik sayısı (sqrt/log2/oran) — ağaçlar
        # arası çeşitliliğin ana kaynağı.
        param_grid={
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16, 24],
            "max_features": ["sqrt", "log2", 0.5],
        },
        report_grid={
            "n_estimators": [100],
            "max_depth": [None, 16, 24],
            "max_features": ["sqrt", "log2"],
        },
        quick_grid={
            "n_estimators": [100],
            "max_depth": [None, 16],
            "max_features": ["sqrt"],
        },
        make_estimator=_make_random_forest,
    ),
    ModelSpec(
        name="gradient_boosting",
        display_name="Gradient Boosting",
        # learning_rate: her ağacın katkısının küçültme çarpanı (küçük değer =
        # yavaş ama genelde daha iyi genelleme); max_depth: tek ağacın derinliği
        # (boosting'de sığ ağaçlar tercih edilir, 1 = karar kütüğü/stump).
        param_grid={
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_depth": [1, 2, 3, 5],
        },
        report_grid={
            "learning_rate": [0.05, 0.1],
            "max_depth": [1, 3],
        },
        quick_grid={
            "learning_rate": [0.05, 0.1],
            "max_depth": [1, 3],
        },
        make_estimator=_make_gradient_boosting,
        # class_weight parametresi olmadığı için denge örnek ağırlığıyla sağlanır.
        fit_with_sample_weight=True,
    ),
]


def _splits_for(corpus: str, manifest: str):
    """Tek corpus için Ödev 1 ile birebir aynı konuşmacı-bağımsız bölmeleri üretir.

    Aynı seed (42) ve aynı "speaker" bölme stratejisi kullanıldığı için train/
    val/test kümeleri Ödev 1'dekiyle özdeştir; bu, KNN ile Ödev 2 modellerinin
    sonuçlarını doğrudan karşılaştırılabilir kılan kritik ayrıntıdır.
    """
    cfg = Config()
    cfg.data.manifest = manifest
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = "speaker"
    cfg.train.seed = SEED
    df = pd.read_csv(manifest)
    return prepare_splits(df, cfg.data, SEED)


def _fit_pca(Xtr_scaled: np.ndarray, requested_dim: int) -> PCA | None:
    """PCA'yı yalnızca verilen fit matrisine uydurur; sıfır boyut "PCA yok" demektir.

    Bileşen sayısı, matematiksel üst sınırı (min(örnek sayısı, öznitelik
    sayısı)) aşamayacağı için gerekirse kırpılır. Validation/test verisi fit'e
    asla katılmaz — bilgi sızıntısı (leakage) engellenir.
    """
    if requested_dim == 0:
        return None
    n_eff = min(requested_dim, Xtr_scaled.shape[1], Xtr_scaled.shape[0])
    return PCA(n_components=n_eff, random_state=SEED).fit(Xtr_scaled)


def _transform_with_pca(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    requested_dim: int,
) -> tuple[np.ndarray, np.ndarray, int | str]:
    """PCA'yı train'e fit edip train ve validation'ı AYNI dönüşümle projekte eder.

    Dönüş değerinin üçüncü elemanı raporlama içindir: PCA kullanılmadıysa
    "none" metni, kullanıldıysa fiilen elde edilen bileşen sayısı döner.
    """
    pca = _fit_pca(Xtr, requested_dim)
    if pca is None:
        return Xtr, Xva, "none"
    return pca.transform(Xtr), pca.transform(Xva), int(pca.n_components_)


def _fit_estimator(spec: ModelSpec, params: dict[str, Any], X: np.ndarray, y: np.ndarray):
    """Modeli kurar ve gerekiyorsa dengeli örnek ağırlıklarıyla fit eder.

    Tree/Forest sınıf dengesizliğini kendi `class_weight="balanced"` ayarıyla
    çözer. HistGradientBoosting'de bu parametre olmadığından aynı etki
    `compute_sample_weight` ile örnek başına ağırlık verilerek elde edilir:
    az örnekli sınıfın her örneği eğitim kaybında daha ağır sayılır.
    """
    estimator = spec.make_estimator(params)
    if spec.fit_with_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Metrik sözlüğünü CSV için prefix'li ve dört ondalığa yuvarlanmış satıra çevirir.

    Prefix ("val" veya "test") sayesinde aynı dört metrik, tabloda hangi kümeye
    ait olduğu belli olacak şekilde sütun adlarına yazılır.
    """
    return {
        f"{prefix}_accuracy": round(metrics["accuracy"], 4),
        f"{prefix}_balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        f"{prefix}_macro_f1": round(metrics["macro_f1"], 4),
        f"{prefix}_weighted_f1": round(metrics["weighted_f1"], 4),
    }


def run_model_for_dataset(
    corpus: str,
    spec: ModelSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_dir: str,
    out_dir: Path,
    grid_mode: str = "report",
) -> dict[str, Any]:
    """Tek corpus-model çifti için geçerleme taramasını ve final testi çalıştırır.

    Ödev 1'deki KNN protokolünün aynısı, farklı bir modelle: feature havuzu (F),
    PCA boyutu (P) ve modelin kendi hiperparametreleri validation kümesinde
    taranır. En iyi ayar train+validation üzerinde yeniden eğitilir; test kümesi
    yalnızca final metrikleri, sınıf raporunu ve karmaşıklık matrisini üretmek
    için BİR kez kullanılır.
    """
    pools, pca_dims, params_grid = _grid_settings(spec, grid_mode)

    rows: list[dict[str, Any]] = []      # tüm denemelerin sonuçları (CSV'ye)
    best: dict[str, Any] | None = None   # şimdiye dek görülen en iyi ayar

    # ================= 1. AŞAMA: VALIDATION ÜZERİNDE IZGARA ARAMASI =========
    for pool in pools:
        # F: mean(768), mean+std(1536) veya mean+std+max(2304).
        # Öznitelikler pool başına bir kez yüklenir; içteki döngüler paylaşır.
        Xtr, ytr = load_pooled(train_df, pool, cache_dir=cache_dir)
        Xva, yva = load_pooled(val_df, pool, cache_dir=cache_dir)
        if len(Xtr) == 0 or len(Xva) == 0:
            log.warning("[%s/%s/%s] missing train or validation features", corpus, spec.name, pool)
            continue

        # Scaler ve PCA validation'a değil, YALNIZCA train'e fit edilir; aksi
        # hâlde validation bilgisi ön işleme üzerinden modele sızardı.
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        for requested_pca in pca_dims:
            # Aynı ölçeklenmiş matris farklı PCA boyutlarında yeniden kullanılır
            # (ölçekleme her PCA denemesi için tekrar hesaplanmaz — zaman kazancı).
            Xtr_p, Xva_p, effective_pca = _transform_with_pca(Xtr_s, Xva_s, requested_pca)

            for params in params_grid:
                # Her kombinasyon için estimator SIFIRDAN kurulur; böylece
                # denemeler birbirinin durumundan etkilenmez.
                estimator = _fit_estimator(spec, params, Xtr_p, ytr)
                metrics = compute_metrics(yva, estimator.predict(Xva_p))
                row = {
                    "corpus": corpus,
                    "model": spec.display_name,
                    "model_key": spec.name,
                    "feature": pool,
                    "feature_dim": POOL_DIM[pool],
                    "pca_requested": "none" if requested_pca == 0 else requested_pca,
                    "pca_dim": effective_pca,
                    "params": _params_to_text(params),
                    **_metric_row("val", metrics),
                }
                rows.append(row)

                # Seçim ölçütü bir üçlü (tuple): birincil makro-F1, eşitlikte
                # dengeli doğruluk, o da eşitse doğruluk. Python tuple'ları
                # eleman eleman karşılaştırdığı için tek `>` yeterlidir.
                score = (
                    metrics["macro_f1"],
                    metrics["balanced_accuracy"],
                    metrics["accuracy"],
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "pool": pool,
                        "requested_pca": requested_pca,
                        "effective_pca": effective_pca,
                        "params": params,
                        "val_metrics": metrics,
                    }

    # Hiçbir kombinasyon çalışmadıysa (ör. cache boşsa) anlaşılır bir hata ver.
    if best is None:
        raise RuntimeError(
            f"No valid validation result for corpus={corpus}, model={spec.name}. "
            "Run odev1/extract.py first or pass a manifest covered by the feature cache."
        )

    # Tüm arama sonuçları, en iyiler üstte olacak şekilde CSV'ye yazılır.
    grid = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_balanced_accuracy", "val_accuracy"], ascending=False
    )
    grid_path = out_dir / f"{spec.name}_validation_grid.csv"
    grid.to_csv(grid_path, index=False)

    # ================= 2. AŞAMA: FİNAL EĞİTİM + TEK SEFERLİK TEST ===========
    # Model seçimi bittiği için validation'ı final eğitim verisine katabiliriz
    # (daha fazla eğitim örneği genellikle daha iyi final model demektir).
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    Xfit, yfit = load_pooled(fit_df, best["pool"], cache_dir=cache_dir)
    Xte, yte = load_pooled(test_df, best["pool"], cache_dir=cache_dir)
    if len(Xfit) == 0 or len(Xte) == 0:
        raise RuntimeError(
            f"No valid train+val or test features for corpus={corpus}, model={spec.name}."
        )

    # Final scaler/PCA, daha büyük train+val verisi üzerinde BAŞTAN fit edilir;
    # test verisi yalnızca transform edilir (fit'e asla katılmaz).
    scaler = StandardScaler().fit(Xfit)
    Xfit_s = scaler.transform(Xfit)
    Xte_s = scaler.transform(Xte)
    pca = _fit_pca(Xfit_s, best["requested_pca"])
    if pca is None:
        Xfit_p, Xte_p = Xfit_s, Xte_s
        test_pca_dim: int | str = "none"
    else:
        Xfit_p, Xte_p = pca.transform(Xfit_s), pca.transform(Xte_s)
        test_pca_dim = int(pca.n_components_)

    estimator = _fit_estimator(spec, best["params"], Xfit_p, yfit)
    test_metrics = compute_metrics(yte, estimator.predict(Xte_p))

    # ---- Sonuç paketi: raporda ve karşılaştırma tablolarında kullanılır ----
    result = {
        "corpus": corpus,
        "model": spec.display_name,
        "model_key": spec.name,
        "best_config": {
            "feature": best["pool"],
            "feature_dim": POOL_DIM[best["pool"]],
            "pca_dim": test_pca_dim,
            "params": best["params"],
            "val_macro_f1": round(best["val_metrics"]["macro_f1"], 4),
            "val_balanced_accuracy": round(best["val_metrics"]["balanced_accuracy"], 4),
        },
        "test": {
            k: round(test_metrics[k], 4)
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "test_per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "validation_grid": str(grid_path),
    }

    result_path = out_dir / f"{spec.name}_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_confusion(
        test_metrics["confusion_matrix"],
        out_dir / f"{spec.name}_confusion_matrix.png",
        title=f"{corpus} - {spec.display_name} (test)",
    )

    log.info(
        "[%s] %s best feat=%s(%d) pca=%s params=%s | val macroF1=%.3f -> test macroF1=%.3f",
        corpus,
        spec.display_name,
        best["pool"],
        POOL_DIM[best["pool"]],
        test_pca_dim,
        _params_to_text(best["params"]),
        best["val_metrics"]["macro_f1"],
        test_metrics["macro_f1"],
    )
    return result


def run_dataset(
    corpus: str,
    manifest: str,
    cache_dir: str,
    out_root: str,
    grid_mode: str = "report",
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Bir corpus'u train/val/test olarak böler ve seçilen model ailelerini sırayla çalıştırır.

    ``model_keys`` None ise üç modelin tamamı koşulur; verilirse yalnızca
    istenen alt küme çalışır (örneğin sadece "random_forest"). Bilinmeyen model
    adları, sessizce atlanmak yerine baştan hata ile reddedilir.
    """
    set_seed(SEED)
    out_dir = ensure_dir(Path(out_root) / corpus)
    train_df, val_df, test_df = _splits_for(corpus, manifest)
    log.info("[%s] train=%d val=%d test=%d", corpus, len(train_df), len(val_df), len(test_df))

    # Model seçimi: ya hepsi ya da kullanıcının istediği doğrulanmış alt küme.
    if model_keys is None:
        specs = MODEL_SPECS
    else:
        known = {s.name for s in MODEL_SPECS}
        unknown = sorted(set(model_keys) - known)
        if unknown:
            raise ValueError(f"Unknown model keys: {unknown}. Expected one of {sorted(known)}.")
        specs = [s for s in MODEL_SPECS if s.name in set(model_keys)]

    results: dict[str, Any] = {}
    for spec in specs:
        results[spec.name] = run_model_for_dataset(
            corpus=corpus,
            spec=spec,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cache_dir=cache_dir,
            out_dir=out_dir,
            grid_mode=grid_mode,
        )
    return results


def _comparison_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """İç içe sonuç sözlüklerini model karşılaştırma CSV'si için düz satırlara açar.

    Girdi yapısı {corpus → {model → sonuç}} şeklindedir; CSV ise satır başına
    bir (corpus, model) çifti ister. Bu fonksiyon o düzleştirmeyi yapar ve her
    satıra en iyi ayar + validation + test metriklerini koyar.
    """
    rows: list[dict[str, Any]] = []
    for corpus, corpus_results in results.items():
        for result in corpus_results.values():
            bc = result["best_config"]
            test = result["test"]
            rows.append(
                {
                    "corpus": corpus,
                    "model": result["model"],
                    "feature": bc["feature"],
                    "feature_dim": bc["feature_dim"],
                    "pca_dim": bc["pca_dim"],
                    "params": _params_to_text(bc["params"]),
                    "val_macro_f1": bc["val_macro_f1"],
                    "val_balanced_accuracy": bc["val_balanced_accuracy"],
                    "test_accuracy": test["accuracy"],
                    "test_balanced_accuracy": test["balanced_accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "test_weighted_f1": test["weighted_f1"],
                }
            )
    return rows


def _load_knn_rows(corpora: tuple[str, ...], knn_out_root: str) -> list[dict[str, Any]]:
    """Ödev 1 KNN JSON çıktısını Ödev 2 karşılaştırma tablosunun şemasına dönüştürür.

    Amaç: KNN'in de aynı tabloda satır olarak yer alması. KNN sonucunda
    balanced accuracy validation için kaydedilmediğinden o hücre boş bırakılır;
    K değeri diğer modellerin hiperparametreleriyle aynı biçimde ("params"
    sütununda JSON olarak) gösterilir. Dosyası olmayan corpus sessizce atlanır.
    """
    rows: list[dict[str, Any]] = []
    for corpus in corpora:
        path = Path(knn_out_root) / corpus / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        bc = result["best_config"]
        test = result["test"]
        rows.append(
            {
                "corpus": corpus,
                "model": "KNN (Odev 1)",
                "feature": bc.get("feature"),
                "feature_dim": bc.get("feature_dim"),
                "pca_dim": bc.get("pca_dim"),
                "params": _params_to_text({"K": bc.get("K")}),
                "val_macro_f1": bc.get("val_macro_f1"),
                "val_balanced_accuracy": "",
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
            }
        )
    return rows


def run_all(
    manifest: str = "data/processed/manifest.csv",
    cache_dir: str = "odev1/cache/w2v",
    out_root: str = "odev2/outputs",
    corpora: tuple[str, ...] = (CORPUS_CREMAD, CORPUS_MELD),
    quick: bool = False,
    grid_mode: str = "report",
    model_keys: tuple[str, ...] | None = None,
    knn_out_root: str = "odev1/outputs",
) -> dict[str, Any]:
    """Tüm corpus/model deneylerini yönetir ve ortak CSV/summary çıktılarını yazar.

    Ödev 2'nin en üst seviye fonksiyonudur. Üretilen dosyalar:
      * ``model_comparison.csv``          — yalnızca Ödev 2 modelleri,
      * ``test_comparison_with_knn.csv``  — Ödev 1 KNN sonucu da eklenmiş hâli,
      * ``summary.json``                  — deney ayarları + tüm sonuçların özeti.

    ``quick=True`` pratik bir kısayoldur: grid_mode'u "quick"e zorlar.
    """
    if quick:
        grid_mode = "quick"
    # Geçersiz mod, saatler süren deney başlamadan önce yakalanır.
    if grid_mode not in GRID_MODES:
        raise ValueError(f"Unknown grid mode: {grid_mode}. Expected one of {GRID_MODES}.")
    out_root_path = ensure_dir(out_root)
    results: dict[str, dict[str, Any]] = {}

    # Her corpus bağımsız çalışır; sonuçlar corpus adıyla toplanır.
    for corpus in corpora:
        results[corpus] = run_dataset(
            corpus, manifest, cache_dir, out_root, grid_mode=grid_mode, model_keys=model_keys
        )

    # ---- Karşılaştırma tabloları -------------------------------------------
    # Önce yalnızca Ödev 2 modelleri; corpus içinde test makro-F1'e göre sıralı.
    comparison = pd.DataFrame(_comparison_rows(results))
    comparison = comparison.sort_values(["corpus", "test_macro_f1"], ascending=[True, False])
    comparison.to_csv(out_root_path / "model_comparison.csv", index=False)

    # Sonra KNN satırları da eklenmiş genişletilmiş tablo (varsa).
    with_knn_rows = _comparison_rows(results) + _load_knn_rows(corpora, knn_out_root)
    if with_knn_rows:
        with_knn = pd.DataFrame(with_knn_rows)
        with_knn = with_knn.sort_values(["corpus", "test_macro_f1"], ascending=[True, False])
        with_knn.to_csv(out_root_path / "test_comparison_with_knn.csv", index=False)

    # ---- summary.json: deney hangi ayarlarla koştu + tüm sonuçlar ----------
    summary = {
        "manifest": manifest,
        "cache_dir": cache_dir,
        "quick": grid_mode == "quick",
        "grid_mode": grid_mode,
        "models": list(model_keys) if model_keys is not None else [s.name for s in MODEL_SPECS],
        "per_dataset": results,
    }
    (out_root_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote comparison tables under %s", out_root_path)
    return summary
