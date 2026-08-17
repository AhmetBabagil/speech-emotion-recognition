"""Klasik taban modeli: MFCC özet istatistikleri -> StandardScaler -> sınıflandırıcı.

Proje önerisinin şart koştuğu derin-olmayan referans noktası budur. Amaç: derin
modellerin katkısını ölçebilmek için önce "eski usul" güçlü bir çizgi çekmek.
Üç sınıflandırıcı seçeneği vardır:
  * "svm"      : RBF çekirdekli SVM (havuzlanmış MFCC öznitelikleri için güçlü
                 ve yerleşik bir varsayılan)
  * "logreg"   : multinomial lojistik regresyon (hızlı, doğrusal — özellikle
                 çok kez eğitim gerektiren analizlerde tercih edilir)
  * "rf"       : rastgele orman (ağaç tabanlı, doğrusal olmayan alternatif)

Sınıf dengesizliği (özellikle MELD'de) her seçenekte ``class_weight="balanced"``
ile ele alınır: nadir sınıfın örnekleri kayıpta daha ağır sayılır.

Neden Pipeline? Ölçekleyici ile sınıflandırıcıyı tek nesnede birleştirmek,
"scaler'ı yalnızca train ile fit et, test'e yalnızca transform uygula" kuralını
otomatikleştirir — test istatistiklerinin modele sızması (leakage) yapısal
olarak imkânsız hâle gelir.
"""

from __future__ import annotations


def build_baseline(kind: str = "svm"):
    """Fit edilmemiş bir sklearn Pipeline'ı (scaler + sınıflandırıcı) döndürür.

    "Fit edilmemiş" olması bilinçli: çağıran taraf hangi veriyle eğiteceğine
    kendisi karar verir (tam veri, etiket kesiri, pseudo-label'lı küme...).
    sklearn importları fonksiyon içindedir ki modül, sklearn kurulu olmayan
    ortamlarda da import edilebilsin.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    kind = kind.lower()
    if kind == "svm":
        from sklearn.svm import SVC

        # probability=True YOK: taban modeli yalnızca .predict() çağırır
        # (bu bayrak sklearn 1.9'da zaten deprecate edildi; ayrıca pahalı bir iç
        # çapraz doğrulama gerektirirdi). Argmax etiketi için karar fonksiyonu yeter.
        # C=10: hatalara görece az tolerans (daha esnek sınır); gamma="scale":
        # çekirdek genişliğini öznitelik varyansından otomatik türet.
        clf = SVC(C=10.0, kernel="rbf", gamma="scale",
                  class_weight="balanced", random_state=42)
    elif kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        # max_iter=2000: 240 boyutlu veride varsayılan 100 iterasyon
        # yakınsamaya yetmeyebilir; uyarı yerine bol iterasyon tanınır.
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        # n_estimators=400 ağaç, n_jobs=-1 ile tüm çekirdeklerde paralel eğitim.
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown baseline kind={kind!r}")

    # Sıra önemli: önce standardizasyon (SVM/logreg ölçeğe duyarlıdır),
    # sonra sınıflandırıcı. Pipeline fit/predict'te bu sırayı otomatik uygular.
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
