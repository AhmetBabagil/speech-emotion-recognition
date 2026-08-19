# Metrikler, karışıklık matrisi (confusion matrix) çizimi ve torch değerlendirme döngüsü.
#
# Bütün metrikler kanonik altı sınıflık etiket uzayı üzerinden hesaplanır; böylece korpus-içi ve korpuslar-arası sonuçlar doğrudan karşılaştırılabilir olur.
#
# Neden bu metrikler?
# * accuracy            : en sezgisel metrik, ama dengesiz veride yanıltıcıdır
# (hep "neutral" diyen bir model MELD'de yüksek accuracy alır).
# * balanced_accuracy   : sınıf başına recall'ların ortalaması; dengesizliğe dayanıklı.
# * macro_f1            : her sınıfın F1'inin AĞIRLIKSIZ ortalaması — azınlık
# sınıflara çoğunlukla eşit önem verir; projenin ana metriği.
# * weighted_f1         : sınıf F1'lerinin destek (örnek sayısı) ağırlıklı ortalaması;
# literatürdeki MELD sonuçlarıyla kıyas için raporlanır.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import CANONICAL_EMOTIONS, NUM_CLASSES
from .utils import get_logger, ensure_dir

log = get_logger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    # Gerçek ve tahmin edilen etiketlerden tüm metrikleri tek sözlükte toplar.
    #
    # sklearn importları fonksiyon içinde tutulur ki modül, sklearn kurulu olmayan ortamlarda da import edilebilsin (hafif bağımlılık ilkesi).
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
        balanced_accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        # Boş dizide sklearn kafa karıştırıcı hatalar verir; erken ve net başarısız ol.
        raise ValueError(f"Cannot compute metrics on empty arrays "
                         f"(y_true={len(y_true)}, y_pred={len(y_pred)}).")
    # labels parametresini AÇIKÇA vermek kritik: test kümesinde hiç örneği
    # olmayan bir sınıf bile matriste/skorlarda yerini korur; böylece
    # karışıklık matrisi her zaman 6x6 olur ve indeksler duygularla hizalanır.
    labels = list(range(NUM_CLASSES))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # float(...)/int(...) dönüşümleri: numpy skalerleri JSON'a yazılamaz;
    # önce saf Python tiplerine çevrilir.
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        # Sınıf bazında ayrıntı: hangi duygu kolay, hangisi zor görülebilsin.
        "per_class": {
            CANONICAL_EMOTIONS[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, out_path, *, title: str = "Confusion matrix", normalize: bool = True):
    # Karışıklık matrisini ısı haritası (heatmap) olarak PNG'ye kaydeder.
    #
    # normalize=True iken her SATIR kendi toplamına bölünür: hücre (i, j), "gerçek sınıfı i olanların yüzde kaçı j tahmin edildi" anlamına gelir. Bu, sınıf boyutları eşit olmadığında ham sayılardan çok daha okunaklıdır.
    import matplotlib
    # "Agg" arka ucu: ekran/pencere gerektirmeden dosyaya çizim yapar; sunucuda
    # ya da testlerde GUI olmadan da çalışması için şarttır.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # np.divide + where: satır toplamı 0 ise (o sınıftan hiç örnek yoksa)
        # 0'a bölme uyarısı yerine satırı 0 olarak bırak.
        cm_disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        fmt = ".2f"   # oranlar için 2 ondalık
    else:
        cm_disp = cm
        fmt = ".0f"   # ham sayılar için tam sayı görünümü
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CANONICAL_EMOTIONS, yticklabels=CANONICAL_EMOTIONS,
                vmin=0, vmax=1 if normalize else None, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()  # figürü kapat: döngüde çağrılırsa bellek sızıntısı olmasın


def report(y_true, y_pred, out_dir, prefix: str = "test", title: str | None = None) -> dict:
    # Tek çağrıda tam raporlama: metrikleri hesapla, JSON + PNG yaz, özet logla.
    #
    # Eğitim betiklerinin her seferinde aynı üç adımı tekrarlamaması için bu "kolaylık" fonksiyonu vardır. Dosya adları ``prefix`` ile başlar (örn. test_metrics.json), böylece aynı klasöre val/test raporları birlikte yazılabilir.
    out_dir = ensure_dir(out_dir)
    metrics = compute_metrics(y_true, y_pred)
    with open(Path(out_dir) / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        Path(out_dir) / f"{prefix}_confusion_matrix.png",
        title=title or f"{prefix} confusion matrix",
    )
    log.info("[%s] acc=%.4f  bal_acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
             prefix, metrics["accuracy"], metrics["balanced_accuracy"],
             metrics["macro_f1"], metrics["weighted_f1"])
    return metrics


def evaluate_torch(model, loader, device):
    # Modeli ``loader`` üzerinde çalıştırır -> (y_true, y_pred, y_prob) numpy dizileri.
    #
    # Değerlendirmede iki önemli ayrıntı:
    # * ``model.eval()``: Dropout kapanır, BatchNorm eğitimde biriktirdiği
    # istatistikleri kullanır — yoksa her değerlendirme farklı sonuç verirdi.
    # * ``torch.no_grad()``: gradyan hesaplanmaz; bellek ve zaman tasarrufu.
    # Olasılıklar (softmax çıktısı) da döndürülür; güven analizi ya da yarı-denetimli pseudo-label seçimi gibi ileri kullanımlar için hazırdır.
    import torch

    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            # non_blocking: pin_memory ile birlikte GPU'ya asenkron kopya sağlar.
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)   # logit -> olasılık dağılımı
            preds.append(prob.argmax(1).cpu().numpy())  # en olası sınıf
            probs.append(prob.cpu().numpy())
            ys.append(np.asarray(yb))
    # Batch listelerini tek büyük diziye birleştir.
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))
