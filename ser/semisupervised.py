"""Denetimsiz (öznitelik kümeleme) ve yarı-denetimli (azaltılmış etiket) analizler.

Proje önerisinde söz verilen bileşeni gerçekler:
"azaltılmış etiketle eğitim ve öznitelik kümeleme üzerinden yarı-denetimli/
denetimsiz bir bileşen" — yani öznitelik kümeleme ve az etiketle eğitim
üzerinden, duygu yapısının ne kadarının akustikte kendiliğinden var olduğunu
ve görevin gerçekte kaç etikete ihtiyaç duyduğunu araştırır.

Üç analiz de MFCC-istatistik özniteliklerini kullanır (klasik taban modelinin
kullandığı 240 boyutlu vektörlerin aynısı). Neden? Böylece diskteki öznitelik
önbelleği yeniden kullanılır ve her şey CPU'da hızlıca koşar; derin model
eğitmeye gerek kalmaz.

  1. cluster_analysis  — Standartlaştırılmış MFCC öznitelikleri üzerinde K-Means
     (K = duygu sayısı). Tamamen DENETİMSİZ: kümeleri öğrenirken hiçbir etiket
     kullanılmaz. Küme kalitesi gerçek duygularla iki şekilde ölçülür:
     Adjusted Rand Index (ARI) ve Normalized Mutual Information (NMI) — ikisi de
     küme numaralarının permütasyonundan bağımsızdır; ayrıca Macar (Hungarian)
     eşleştirmesiyle küme→duygu ataması yapılıp accuracy / makro-F1 raporlanır
     (yorumlanabilir bir "etiketsiz sınıflandırma" skoru).

  2. label_efficiency  — Taban modelini etiketlerin yalnızca bir kesiriyle
     (%1 … %100) eğitir ve test makro-F1'ini raporlar (öğrenme eğrisi).
     Etiket azaldıkça performansın nasıl düştüğünü gösterir.

  3. self_training     — YARI-DENETİMLİ pseudo-labeling (kendi kendini eğitme):
     küçük bir etiketli çekirdekten başlayıp, etiketsiz havuzdan yüksek güvenli
     pseudo-etiketleri yinelemeli olarak ekler ve modeli yeniden eğitir.
     Adil kıyas için AYNI etiket bütçesiyle eğitilmiş salt-denetimli modelle
     karşılaştırılır: etiketsiz veri gerçekten işe yarıyor mu?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .constants import CANONICAL_EMOTIONS, NUM_CLASSES
from .data import mfcc_feature_matrix, prepare_splits
from .evaluate import compute_metrics, save_confusion_matrix
from .models import build_baseline
from .utils import get_logger, set_seed, ensure_dir

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Ortak öznitelik yükleme
# --------------------------------------------------------------------------- #
def _load_features(cfg: Config):
    """(X_train, y_train, X_test, y_test) MFCC-istatistik matrislerini döndürür.

    train+val birleştirilip "etiketli havuz" yapılır (bu analizlerde ayrı bir
    doğrulama kümesine gerek yok); test ise kenarda tutulur. Bölme, projenin
    geri kalanıyla AYNI konuşmacı-bağımsız protokolü kullanır — böylece buradaki
    skorlar diğer deneylerle karşılaştırılabilir kalır.
    """
    df = pd.read_csv(cfg.data.manifest)
    train_df, val_df, test_df = prepare_splits(df, cfg.data, cfg.train.seed)
    pool_df = pd.concat([train_df, val_df], ignore_index=True)
    log.info("Extracting MFCC features (pool=%d, test=%d) ...", len(pool_df), len(test_df))
    X_tr, y_tr = mfcc_feature_matrix(pool_df, cfg)
    X_te, y_te = mfcc_feature_matrix(test_df, cfg)
    return X_tr, y_tr, X_te, y_te


def _stratified_subset(y: np.ndarray, fraction: float, rng, min_per_class: int = 1):
    """Sınıf-katmanlı (stratified) bir alt kümenin indekslerini döndürür.

    "Katmanlı" demek: her sınıftan, o sınıfın büyüklüğüyle orantılı sayıda örnek
    seçilir; böylece küçük kesirlerde bile sınıf dağılımı korunur. Ayrıca her
    mevcut sınıftan EN AZ ``min_per_class`` örnek garanti edilir — yoksa %1 gibi
    kesirlerde nadir bir sınıf tamamen boş kalır ve eğitim o sınıfı hiç göremezdi.
    """
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n = int(round(fraction * len(c_idx)))
        # Alt sınır: min_per_class; üst sınır: sınıftaki örnek sayısı.
        n = max(min_per_class, min(n, len(c_idx)))
        idx.extend(rng.choice(c_idx, size=n, replace=False).tolist())
    # sorted: indeks sırası deterministik olsun (aynı seed -> aynı alt küme).
    return np.array(sorted(idx))


# --------------------------------------------------------------------------- #
# 1. Denetimsiz kümeleme
# --------------------------------------------------------------------------- #
def cluster_analysis(cfg: Config, out_dir: Path, n_clusters: int | None = None) -> dict:
    """K-Means kümelerinin gerçek duygularla ne kadar örtüştüğünü ölçer.

    Sorduğu soru: "Etiketlere hiç bakmadan, akustik öznitelikler duygulara göre
    kendiliğinden öbekleniyor mu?" Yanıt ARI/NMI (etiket-permütasyonundan
    bağımsız uyum ölçüleri) ve Hungarian-eşleşmeli accuracy ile verilir.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment

    out_dir = ensure_dir(out_dir)
    k = n_clusters or NUM_CLASSES  # varsayılan: duygu sayısı kadar küme
    X_tr, y_tr, X_te, y_te = _load_features(cfg)

    # Standartlaştırma K-Means için önemlidir (Öklid mesafesi ölçeğe duyarlı).
    # Scaler yalnızca TRAIN üzerinde fit edilir: test istatistikleri modele
    # sızmasın (leakage önlenir), sonra iki kümeye de aynı dönüşüm uygulanır.
    scaler = StandardScaler().fit(X_tr)
    Xtr, Xte = scaler.transform(X_tr), scaler.transform(X_te)

    # n_init=10: K-Means yerel minimuma takılabilir; 10 farklı başlangıçtan
    # en iyisi seçilir. random_state ile sonuç tekrarlanabilir.
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.train.seed).fit(Xtr)
    c_tr, c_te = km.labels_, km.predict(Xte)

    # Küme numaraları keyfîdir (küme 0'ın hangi duygu olduğu belirsiz). Küme id
    # -> duygu id eşlemesi, TRAIN ortak-görülme (co-occurrence) matrisi üzerinde
    # Macar algoritmasıyla yapılır (toplam uyumu MAKSİMİZE eden birebir atama;
    # linear_sum_assignment minimizasyon yaptığı için matrise eksi işareti
    # konur). Bu SABİT eşleme sonra test kümelerine uygulanır — eşlemeyi test
    # üzerinde seçmek gizli bir bilgi sızıntısı olurdu.
    co = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, t in zip(c_tr, y_tr):
        co[cl, t] += 1
    rows, cols = linear_sum_assignment(-co)
    cluster_to_label = {int(r): int(c) for r, c in zip(rows, cols)}
    # k > sınıf sayısı ise bazı kümeler eşlenmeden kalır; onlar da en sık
    # gördükleri duyguya (satırın argmax'ı) atanır.
    for cl in range(k):
        cluster_to_label.setdefault(cl, int(co[cl].argmax()))
    y_pred = np.array([cluster_to_label[int(cl)] for cl in c_te])

    metrics = compute_metrics(y_te, y_pred)
    # ARI/NMI ham küme numaraları üzerinden hesaplanır (eşleme gerektirmez).
    ari = float(adjusted_rand_score(y_te, c_te))
    nmi = float(normalized_mutual_info_score(y_te, c_te))

    result = {
        "n_clusters": k,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "hungarian_accuracy": metrics["accuracy"],
        "hungarian_macro_f1": metrics["macro_f1"],
        # Hangi kümenin hangi duyguya atandığı — nitel yorum için faydalı.
        "cluster_to_emotion": {int(c): CANONICAL_EMOTIONS[lbl] for c, lbl in cluster_to_label.items()},
    }
    with open(out_dir / "cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    save_confusion_matrix(
        metrics["confusion_matrix"], out_dir / "cluster_confusion.png",
        title="Unsupervised K-Means (Hungarian-matched)", normalize=True,
    )
    # Loga şans seviyesi (1/6) de yazılır: skorun bağlamı olsun.
    log.info("[cluster] ARI=%.3f NMI=%.3f hungarian_acc=%.3f macroF1=%.3f (chance acc=%.3f)",
             ari, nmi, metrics["accuracy"], metrics["macro_f1"], 1.0 / NUM_CLASSES)
    return result


# --------------------------------------------------------------------------- #
# 2. Azaltılmış etiketle öğrenme eğrisi
# --------------------------------------------------------------------------- #
# Log ölçekte yayılmış kesirler: az-etiket bölgesini (%1-%10) ince, bol-etiket
# bölgesini kaba örnekler — eğrinin ilginç kısmı zaten soldadır.
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)


def label_efficiency(cfg: Config, out_dir: Path, fractions=DEFAULT_FRACTIONS,
                     kind: str = "logreg") -> list[dict]:
    """Her etiket kesiri için taban modeli eğitir, test skorlarını toplar.

    Varsayılan sınıflandırıcı logreg'dir çünkü hızlıdır ve bu deneyde amaç
    mutlak en iyi skor değil, kesirler arasındaki EĞİLİMİ görmektir.
    """
    out_dir = ensure_dir(out_dir)
    X_tr, y_tr, X_te, y_te = _load_features(cfg)
    rng = np.random.default_rng(cfg.train.seed)

    rows = []
    for f in fractions:
        # Her kesirde sınıf dağılımını koruyan bir alt küme seç.
        idx = _stratified_subset(y_tr, f, rng)
        # Her kesir için SIFIRDAN yeni bir pipeline kur (scaler dahil):
        # önceki kesirin öğrendikleri sonraki ölçüme karışmasın.
        pipe = build_baseline(kind)
        pipe.fit(X_tr[idx], y_tr[idx])
        m = compute_metrics(y_te, pipe.predict(X_te))
        rows.append({"label_fraction": float(f), "n_labeled": int(len(idx)),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        log.info("[label-eff] frac=%.2f n=%d acc=%.3f macroF1=%.3f",
                 f, len(idx), m["accuracy"], m["macro_f1"])

    pd.DataFrame(rows).to_csv(out_dir / "label_efficiency.csv", index=False)
    _plot_label_curve(rows, out_dir / "label_efficiency.png")
    return rows


def _plot_label_curve(rows, out_path):
    """Öğrenme eğrisini çizer: x = etiket yüzdesi (log ölçek), y = test skoru."""
    import matplotlib
    matplotlib.use("Agg")  # GUI'siz çizim (dosyaya)
    import matplotlib.pyplot as plt

    fr = [r["label_fraction"] * 100 for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(fr, f1, "o-", label="macro-F1")
    plt.plot(fr, acc, "s--", label="accuracy")
    # Şans çizgisi (1/6): skorların anlamlı olup olmadığını gösteren referans.
    plt.axhline(1.0 / NUM_CLASSES, color="gray", ls=":", label="chance")
    # Log ölçek: %1 ile %10 arasındaki fark, %50 ile %100 arasındaki kadar yer kaplasın.
    plt.xscale("log")
    plt.xlabel("Labeled fraction of training set (%, log scale)")
    plt.ylabel("Test score")
    plt.title("Label efficiency (reduced-label training)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 3. Yarı-denetimli kendi-kendini-eğitme (self-training)
# --------------------------------------------------------------------------- #
def self_training(cfg: Config, out_dir: Path, label_fraction: float = 0.10,
                  threshold: float = 0.80, iterations: int = 10) -> dict:
    """Pseudo-labeling ile yarı-denetimli eğitim ve salt-denetimli kıyası.

    Fikir: az sayıda gerçek etiketle bir model eğit; etiketsiz havuzda modelin
    ÇOK emin olduğu (olasılık >= threshold) tahminleri "sanki gerçek etiketmiş
    gibi" eğitim kümesine ekle; modeli yeniden eğit; tekrarla. Eşik yüksek
    tutulur çünkü yanlış pseudo-etiketler hatayı besleyip büyütür
    (confirmation bias) — kaliteli az örnek, bol gürültülü örnekten iyidir.
    """
    out_dir = ensure_dir(out_dir)
    X_tr, y_tr, X_te, y_te = _load_features(cfg)
    rng = np.random.default_rng(cfg.train.seed)

    # Küçük, sınıf-dengeli etiketli çekirdek (örn. etiketlerin %10'u).
    seed_idx = _stratified_subset(y_tr, label_fraction, rng)
    labeled = np.zeros(len(y_tr), dtype=bool)
    labeled[seed_idx] = True

    # Referans: AYNI etiket bütçesiyle salt-denetimli model. Bu kıyas olmadan
    # self-training'in katkısı ölçülemezdi.
    sup = build_baseline("logreg").fit(X_tr[seed_idx], y_tr[seed_idx])
    sup_m = compute_metrics(y_te, sup.predict(X_te))

    # Self-training döngüsü: etiketli kümeyi yüksek güvenli pseudo-etiketlerle büyüt.
    X_lab, y_lab = X_tr[seed_idx].copy(), y_tr[seed_idx].copy()
    pool = ~labeled          # etiketsiz havuzun maskesi
    n_pseudo = 0
    for _ in range(iterations):
        clf = build_baseline("logreg").fit(X_lab, y_lab)
        if not pool.any():
            break  # havuz tükendi: eklenecek örnek kalmadı
        proba = clf.predict_proba(X_tr[pool])
        conf, pred = proba.max(1), proba.argmax(1)   # her örnek için güven + tahmin
        take = conf >= threshold                     # yalnızca yüksek güvenliler
        if not take.any():
            break  # hiçbir tahmin yeterince güvenli değil: erken dur
        pool_idx = np.where(pool)[0]                 # maske -> gerçek indeksler
        add = pool_idx[take]
        X_lab = np.vstack([X_lab, X_tr[add]])
        # Dikkat: GERÇEK etiket değil, modelin TAHMİNİ eklenir (pseudo-label).
        y_lab = np.concatenate([y_lab, pred[take]])
        pool[add] = False                            # eklenenleri havuzdan düş
        n_pseudo += int(take.sum())

    # Son model: çekirdek + tüm pseudo-etiketlerle bir kez daha eğitilir.
    final = build_baseline("logreg").fit(X_lab, y_lab)
    semi_m = compute_metrics(y_te, final.predict(X_te))

    result = {
        "label_fraction": label_fraction,
        "n_seed_labels": int(len(seed_idx)),
        "n_pseudo_labels_added": n_pseudo,
        "confidence_threshold": threshold,
        "supervised_only": {"accuracy": sup_m["accuracy"], "macro_f1": sup_m["macro_f1"]},
        "self_training": {"accuracy": semi_m["accuracy"], "macro_f1": semi_m["macro_f1"]},
        # Pozitifse etiketsiz veri işe yaramış demektir.
        "macro_f1_delta": semi_m["macro_f1"] - sup_m["macro_f1"],
    }
    with open(out_dir / "self_training.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("[self-train] seed=%d(%.0f%%) +%d pseudo | supervised macroF1=%.3f -> "
             "self-train macroF1=%.3f (Δ=%+.3f)",
             len(seed_idx), label_fraction * 100, n_pseudo,
             sup_m["macro_f1"], semi_m["macro_f1"], result["macro_f1_delta"])
    return result


# --------------------------------------------------------------------------- #
# Sürücü (üç analizi arka arkaya koşan giriş noktası)
# --------------------------------------------------------------------------- #
def run_all(cfg: Config) -> dict:
    """Üç analizi de çalıştırır ve tek bir özet JSON'da toplar."""
    set_seed(cfg.train.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    # Config'i çıktı klasörüne kaydet: bu sonuçlar hangi ayarlarla üretildi,
    # aylar sonra bile bakınca belli olsun.
    cfg.save(out_dir / "config.yaml")
    log.info("=== Unsupervised clustering ===")
    clu = cluster_analysis(cfg, out_dir)
    log.info("=== Reduced-label learning curve ===")
    eff = label_efficiency(cfg, out_dir)
    log.info("=== Semi-supervised self-training ===")
    semi = self_training(cfg, out_dir)
    summary = {"clustering": clu, "label_efficiency": eff, "self_training": semi}
    with open(out_dir / "semisupervised_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
