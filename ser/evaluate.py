# Metrikler, karışıklık matrisi (confusion matrix) çizimi ve torch değerlendirme döngüsü.
#
# Bütün metrikler kanonik altı sınıflık etiket uzayı üzerinden hesaplanır; böylece korpus-içi ve korpuslar-arası sonuçlar doğrudan karşılaştırılabilir olur.
#
# Neden bu metrikler?
# * accuracy            : en sezgisel metrik, ama dengesiz veride yanıltıcıdır (hep "neutral" diyen bir model MELD'de yüksek accuracy alır).
# * balanced_accuracy   : sınıf başına recall'ların ortalaması; dengesizliğe dayanıklı.
# * macro_f1            : her sınıfın F1'inin AĞIRLIKSIZ ortalaması — azınlık sınıflara çoğunlukla eşit önem verir; projenin ana metriği.
# * weighted_f1         : sınıf F1'lerinin destek (örnek sayısı) ağırlıklı ortalaması; literatürdeki MELD sonuçlarıyla kıyas için raporlanır.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import json  # metrikleri JSON'a yazmak için
from pathlib import Path  # dosya yolları

import numpy as np  # diziler

from .constants import CANONICAL_EMOTIONS, NUM_CLASSES  # duygu adları + sınıf sayısı
from .utils import get_logger, ensure_dir  # günlük + klasör

log = get_logger(__name__)  # günlükleyici


def compute_metrics(y_true, y_pred) -> dict:
    # Gerçek ve tahmin edilen etiketlerden tüm metrikleri tek sözlükte toplar.
    #
    # sklearn importları fonksiyon içinde tutulur ki modül, sklearn kurulu olmayan ortamlarda da import edilebilsin (hafif bağımlılık ilkesi).
    from sklearn.metrics import (  # (fonksiyon içi) metrik fonksiyonları
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
        balanced_accuracy_score,
    )

    y_true = np.asarray(y_true)  # gerçek etiketler
    y_pred = np.asarray(y_pred)  # tahminler
    if len(y_true) == 0 or len(y_pred) == 0:  # boş dizi kontrolü
        # Boş dizide sklearn kafa karıştırıcı hatalar verir; erken ve net başarısız ol.
        raise ValueError(f"Cannot compute metrics on empty arrays "  # açık hata
                         f"(y_true={len(y_true)}, y_pred={len(y_pred)}).")
    # labels parametresini AÇIKÇA vermek kritik: test kümesinde hiç örneği
    # olmayan bir sınıf bile matriste/skorlarda yerini korur; böylece
    # karışıklık matrisi her zaman 6x6 olur ve indeksler duygularla hizalanır.
    labels = list(range(NUM_CLASSES))  # 0..5 sabit sınıf listesi
    p, r, f1, support = precision_recall_fscore_support(  # sınıf başına kesinlik/duyarlılık/F1/destek
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # 6x6 karışıklık matrisi
    # float(...)/int(...) dönüşümleri: numpy skalerleri JSON'a yazılamaz;
    # önce saf Python tiplerine çevrilir.
    return {  # tüm metrikleri tek sözlükte topla
        "accuracy": float(accuracy_score(y_true, y_pred)),  # doğruluk
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),  # dengeli doğruluk
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),  # macro-F1
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),  # ağırlıklı F1
        # Sınıf bazında ayrıntı: hangi duygu kolay, hangisi zor görülebilsin.
        "per_class": {  # sınıf-bazı sonuçlar
            CANONICAL_EMOTIONS[i]: {  # her duygu için
                "precision": float(p[i]), "recall": float(r[i]),  # kesinlik + duyarlılık
                "f1": float(f1[i]), "support": int(support[i]),  # F1 + örnek sayısı
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),  # matris (liste olarak)
    }


def save_confusion_matrix(cm, out_path, *, title: str = "Confusion matrix", normalize: bool = True):
    # Karışıklık matrisini ısı haritası (heatmap) olarak PNG'ye kaydeder.
    #
    # normalize=True iken her SATIR kendi toplamına bölünür: hücre (i, j), "gerçek sınıfı i olanların yüzde kaçı j tahmin edildi" anlamına gelir. Bu, sınıf boyutları eşit olmadığında ham sayılardan çok daha okunaklıdır.
    import matplotlib  # grafik kütüphanesi
    # "Agg" arka ucu: ekran/pencere gerektirmeden dosyaya çizim yapar; sunucuda
    # ya da testlerde GUI olmadan da çalışması için şarttır.
    matplotlib.use("Agg")  # ekransız arka uç
    import matplotlib.pyplot as plt  # çizim
    import seaborn as sns  # ısı haritası

    cm = np.asarray(cm, dtype=np.float64)  # matrisi float'a çevir
    if normalize:  # satır-normalize isteniyorsa
        row_sums = cm.sum(axis=1, keepdims=True)  # her satırın toplamı
        # np.divide + where: satır toplamı 0 ise (o sınıftan hiç örnek yoksa)
        # 0'a bölme uyarısı yerine satırı 0 olarak bırak.
        cm_disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)  # oranlar
        fmt = ".2f"   # oranlar için 2 ondalık
    else:  # ham sayılar
        cm_disp = cm  # olduğu gibi
        fmt = ".0f"   # ham sayılar için tam sayı görünümü
    ensure_dir(Path(out_path).parent)  # çıktı klasörünü oluştur
    plt.figure(figsize=(6.5, 5.5))  # figür boyutu
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues",  # ısı haritasını çiz
                xticklabels=CANONICAL_EMOTIONS, yticklabels=CANONICAL_EMOTIONS,
                vmin=0, vmax=1 if normalize else None, cbar=True)
    plt.xlabel("Predicted")  # x ekseni: tahmin
    plt.ylabel("True")  # y ekseni: gerçek
    plt.title(title)  # başlık
    plt.tight_layout()  # yerleşimi sıkılaştır
    plt.savefig(out_path, dpi=150)  # PNG kaydet
    plt.close()  # figürü kapat: döngüde çağrılırsa bellek sızıntısı olmasın


def report(y_true, y_pred, out_dir, prefix: str = "test", title: str | None = None) -> dict:
    # Tek çağrıda tam raporlama: metrikleri hesapla, JSON + PNG yaz, özet logla.
    #
    # Eğitim betiklerinin her seferinde aynı üç adımı tekrarlamaması için bu "kolaylık" fonksiyonu vardır. Dosya adları ``prefix`` ile başlar (örn. test_metrics.json), böylece aynı klasöre val/test raporları birlikte yazılabilir.
    out_dir = ensure_dir(out_dir)  # çıktı klasörünü oluştur
    metrics = compute_metrics(y_true, y_pred)  # metrikleri hesapla
    with open(Path(out_dir) / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:  # metrikleri JSON'a yaz
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(  # karışıklık matrisi PNG'sini yaz
        metrics["confusion_matrix"],
        Path(out_dir) / f"{prefix}_confusion_matrix.png",
        title=title or f"{prefix} confusion matrix",
    )
    log.info("[%s] acc=%.4f  bal_acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f",  # özet logla
             prefix, metrics["accuracy"], metrics["balanced_accuracy"],
             metrics["macro_f1"], metrics["weighted_f1"])
    return metrics  # metrikleri döndür


def evaluate_torch(model, loader, device):
    # Modeli ``loader`` üzerinde çalıştırır -> (y_true, y_pred, y_prob) numpy dizileri.
    #
    # Değerlendirmede iki önemli ayrıntı:
    # * ``model.eval()``: Dropout kapanır, BatchNorm eğitimde biriktirdiği istatistikleri kullanır — yoksa her değerlendirme farklı sonuç verirdi.
    # * ``torch.no_grad()``: gradyan hesaplanmaz; bellek ve zaman tasarrufu.
    # Olasılıklar (softmax çıktısı) da döndürülür; güven analizi ya da yarı-denetimli pseudo-label seçimi gibi ileri kullanımlar için hazırdır.
    import torch  # (fonksiyon içi) torch

    model.eval()  # değerlendirme kipi (Dropout/BatchNorm sabit)
    ys, preds, probs = [], [], []  # gerçek/tahmin/olasılık biriktir
    with torch.no_grad():  # gradyan yok
        for xb, yb in loader:  # her yığın için
            # non_blocking: pin_memory ile birlikte GPU'ya asenkron kopya sağlar.
            xb = xb.to(device, non_blocking=True)  # girdiyi cihaza taşı
            logits = model(xb)  # ham puanlar
            prob = torch.softmax(logits, dim=1)   # logit -> olasılık dağılımı
            preds.append(prob.argmax(1).cpu().numpy())  # en olası sınıf
            probs.append(prob.cpu().numpy())  # olasılıklar
            ys.append(np.asarray(yb))  # gerçek etiketler
    # Batch listelerini tek büyük diziye birleştir.
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))  # (y_true, y_pred, y_prob)
