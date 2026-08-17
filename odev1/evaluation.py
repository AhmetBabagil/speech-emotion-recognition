"""Ödev 1 için metrikler + karmaşıklık matrisi çizimi (numpy/pandas/sklearn + matplotlib).

Bu modül reponun geri kalanından bilerek bağımsız tutulmuştur: böylece KNN
aşaması yalnızca ödevde izin verilen kütüphaneleri kullanır (metrikler için
scikit-learn, çizim için matplotlib — seaborn yok, derin öğrenme kütüphanesi yok).
Ödev 2 de aynı fonksiyonları içe aktararak kullanır; metrik tanımlarının iki
ödevde birebir aynı olması sonuçları doğrudan karşılaştırılabilir kılar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Proje kökünü arama yoluna ekle (doğrudan çalıştırma senaryosu için).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # yalnızca etiket uzayı için  # noqa: E402


def compute_metrics(y_true, y_pred) -> dict:
    """Tahminlerden genel, dengeli ve sınıf bazlı sınıflandırma metrikleri üretir.

    Neden birden çok metrik? Duygu veri setleri dengesizdir (bazı duygular çok
    daha az örneğe sahiptir), tek bir "doğruluk" yanıltıcı olabilir:
      * accuracy           — doğru tahmin oranı; çoğunluk sınıfına eğilimlidir,
      * balanced_accuracy  — her sınıfın recall'unun ortalaması; dengesizlikten etkilenmez,
      * macro_f1           — her sınıfın F1'ine EŞİT ağırlık verir (küçük sınıflar önemli),
      * weighted_f1        — sınıf F1'lerini örnek sayısına (support) göre ağırlıklar.

    Dönen sözlük ayrıca sınıf başına precision/recall/F1/support değerlerini ve
    ham sayılarla karmaşıklık (confusion) matrisini içerir.
    `zero_division=0`: bir sınıf hiç tahmin edilmediğinde hata/uyarı yerine 0 yazılır.
    """
    # sklearn.metrics fonksiyon içinde import edilir: modülün import maliyeti
    # düşük kalır ve bağımlılık yalnızca gerçekten gerektiğinde yüklenir.
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_recall_fscore_support, confusion_matrix,
    )
    # labels'ı açıkça vermek, test kümesinde hiç görünmeyen sınıfların bile
    # tabloda (0 değerleriyle) yer almasını garanti eder.
    labels = list(range(NUM_CLASSES))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        # Sınıf indeksleri okunabilir duygu adlarına çevrilerek raporlanır.
        "per_class": {CANONICAL_EMOTIONS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                              "f1": float(f1[i]), "support": int(sup[i])}
                      for i in labels},
        # .tolist(): numpy dizisi JSON'a yazılabilir saf Python listesine çevrilir.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, out_path, title="Karmaşıklık matrisi", normalize=True):
    """Karmaşıklık matrisini PNG olarak çizer; satır = gerçek sınıf, sütun = tahmin.

    Köşegen üzerindeki değerler doğru tahminlerdir; köşegen dışı hücreler hangi
    duygunun hangisiyle karıştırıldığını gösterir.

    Varsayılan satır normalizasyonu her gerçek sınıfın tahminlere dağılımını
    ORAN olarak gösterir. Bu, sınıflar farklı örnek sayısına (support) sahipken
    karşılaştırmayı kolaylaştırır: ham sayılarla kalabalık sınıf her zaman
    "koyu" görünür ve tablo yanlış okunabilirdi.
    """
    # matplotlib fonksiyon içinde import edilir ve "Agg" arka ucu seçilir:
    # ekran/pencere gerektirmez, sunucuda veya arka planda da PNG üretebilir.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        # Her satırı o gerçek sınıfın toplam örnek sayısına böl.
        # np.divide'ın where= parametresi, toplamı 0 olan (hiç örneği olmayan)
        # satırlarda 0'a bölme hatasını önler; o satırlar 0 olarak kalır.
        rs = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, rs, out=np.zeros_like(cm), where=rs != 0)
        fmt = ".2f"  # oranlar iki ondalıkla gösterilir
    else:
        disp, fmt = cm, ".0f"  # ham sayılar tam sayı olarak gösterilir
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    # Mavi tonlu ısı haritası; normalize modda renk skalası 0-1'e sabitlenir ki
    # farklı grafiklerdeki renkler aynı anlamı taşısın.
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    # Duygu adları eksenlere yazılır; x ekseninde 45 derece döndürülür ki sığsın.
    ax.set_xticklabels(CANONICAL_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(CANONICAL_EMOTIONS)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title)
    # Her hücrenin içine sayısal değeri yaz; koyu hücrede beyaz, açık hücrede
    # siyah yazı kullanılır ki her koşulda okunabilsin.
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, format(disp[i, j], fmt), ha="center", va="center",
                    color="white" if disp[i, j] > (0.5 if normalize else disp.max() / 2) else "black",
                    fontsize=8)
    # fraction/pad: renk çubuğunu grafiğe oranlı ve bitişik boyutlandıran standart değerler.
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)  # belleği serbest bırak (çok grafik üretilirken sızıntıyı önler)
