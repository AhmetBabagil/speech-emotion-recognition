"""Ödev 1 ve Ödev 2'nin depoya işlenmiş deney kanıtlarını DEĞİŞTİRMEDEN yazdırır.

Amaç şeffaflık: değerlendiren kişi (ya da gelecekteki siz) tek komutla
odev1/ ve odev2/ klasörlerindeki resmi CSV/JSON çıktıların özetini görebilsin.
Betik tamamen salt-okunurdur — hiçbir deneyi yeniden koşmaz, hiçbir dosyayı
değiştirmez; yalnızca kayıtlı sonuçları okuyup düzenli tablolar halinde basar.
(Ekrana basılan başlıklar, eski terminallerde bozulmasın diye bilerek Türkçe
karakter içermez; bunlar çıktının bir parçası olduğu için olduğu gibi bırakıldı.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ROOT = proje kökü. Betik nereden çalıştırılırsa çalıştırılsın aşağıdaki tüm
# yollar bu mutlak köke göre kurulur; import için de sys.path'e eklenir.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ser.config import Config  # noqa: E402
from ser.data import prepare_splits  # noqa: E402

# Ödev kanıtlarının depodaki sabit konumları ve incelenen iki korpus.
# Sabitlerin en üstte toplanması, yol değişirse tek yerden güncellenmesini sağlar.
MANIFEST = ROOT / "odev1" / "manifest_subset.csv"
ODEV1_OUT = ROOT / "odev1" / "outputs"
ODEV2_OUT = ROOT / "odev2" / "outputs"
CORPORA = ("cremad", "meld")


def heading(title: str) -> None:
    """Başlığı, altına başlıkla aynı uzunlukta '=' çizgisi çekerek yazdırır.

    Küçük bir görsel ayraç: uzun terminal çıktısında bölümlerin nerede
    başladığı tek bakışta seçilebilsin diye.
    """
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def show_table(frame: pd.DataFrame) -> None:
    """DataFrame'i satır/sütun kırpması olmadan tam genişlikte yazdırır.

    pandas, geniş tabloları varsayılan olarak '...' ile kısaltır. Burada
    `option_context` ile satır/sütun/genişlik sınırlarını GEÇİCİ olarak
    yükseltiyoruz: with bloğu bitince ayarlar eski haline döner, yani global
    pandas yapılandırması kirletilmez. Boş tablo için de sessiz kalmak yerine
    "(veri yok)" basıyoruz ki eksik veri fark edilsin.
    """
    if frame.empty:
        print("(veri yok)")
        return
    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        30,
        "display.width",
        220,
        "display.max_colwidth",
        70,
    ):
        print(frame.to_string(index=False))


def read_json(path: Path) -> dict:
    """JSON dosyasını okur; dosya yoksa açıklayıcı bir hatayla durur.

    Sessizce boş sözlük döndürmek yerine bilerek hata fırlatıyoruz: eksik bir
    çıktı dosyası, kanıt zincirinde bir sorun olduğunun işaretidir ve
    gizlenmemelidir (fail fast ilkesi).
    """
    if not path.exists():
        raise FileNotFoundError(f"Beklenen cikti bulunamadi: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_for(df: pd.DataFrame, corpus: str):
    """Tek bir korpus için konuşmacı-bağımsız train/val/test bölünmesini üretir.

    Kritik nokta: deneylerde kullanılan `prepare_splits` fonksiyonunun TA
    KENDİSİNİ, aynı sabit tohumla (seed=42) çağırır. Böylece burada gösterilen
    bölünme, ödev deneylerinde kullanılanla birebir aynıdır — ayrı bir kopya
    mantık yazsaydık zamanla sessizce farklılaşabilirdi.
    """
    cfg = Config()
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = "speaker"
    return prepare_splits(df, cfg.data, seed=42)


def show_data_statistics() -> None:
    """Manifest'teki kayıt/konuşmacı sayılarını ve split bütünlüğünü gösterir."""
    heading("VERI VE SPLIT ISTATISTIKLERI")
    df = pd.read_csv(MANIFEST)

    # Korpus başına toplam kayıt ve tekil konuşmacı sayısı:
    # "size" grup içindeki satırları sayar, "nunique" tekil değerleri sayar.
    overview = (
        df.groupby("corpus")
        .agg(kayit=("path", "size"), konusmaci=("speaker", "nunique"))
        .reset_index()
    )
    show_table(overview)

    # Korpus x duygu çapraz tablosu: sınıf dengesizliği bir bakışta görülür.
    # (Örn. MELD'de "neutral" başat sınıftır — bu, accuracy yerine macro-F1
    # gibi dengesizliğe duyarlı metrikleri neden raporladığımızı açıklar.)
    heading("SINIF DAGILIMI")
    distribution = pd.crosstab(df["corpus"], df["emotion"]).reset_index()
    show_table(distribution)

    heading("SPEAKER-INDEPENDENT TRAIN / VALIDATION / TEST")
    rows = []
    for corpus in CORPORA:
        # Her korpus için deneylerdekiyle aynı bölünmeyi yeniden üret ve
        # fold başına kayıt/konuşmacı sayılarını tabloya ekle.
        train_df, val_df, test_df = split_for(df, corpus)
        folds = {"train": train_df, "validation": val_df, "test": test_df}
        for name, part in folds.items():
            rows.append(
                {
                    "corpus": corpus,
                    "fold": name,
                    "kayit": len(part),
                    "konusmaci": part["speaker"].nunique(),
                }
            )

        # Kanıtın en önemli satırı: fold'lar arasında konuşmacı kesişimi.
        # Konuşmacı-bağımsız protokolde bu üç sayının da 0 olması ZORUNLUDUR;
        # aksi halde model test konuşmacısının sesini eğitimde "ezberler" ve
        # skorlar şişer (veri sızıntısı). Kümeler kesişimi (&) bunu doğrular.
        speaker_sets = {name: set(part["speaker"].astype(str)) for name, part in folds.items()}
        overlaps = {
            "train-validation": len(speaker_sets["train"] & speaker_sets["validation"]),
            "train-test": len(speaker_sets["train"] & speaker_sets["test"]),
            "validation-test": len(speaker_sets["validation"] & speaker_sets["test"]),
        }
        print(f"{corpus} konusmaci kesisimleri: {overlaps}")
    show_table(pd.DataFrame(rows))


def show_assignment_1() -> None:
    """Ödev 1'in (KNN) final test sonuçlarını ve validasyon taramasını gösterir."""
    heading("ODEV 1 - KNN FINAL TEST SONUCLARI")
    result_rows = []
    for corpus in CORPORA:
        result = read_json(ODEV1_OUT / corpus / "result.json")
        # "best_config": validasyonda en iyi macro-F1'i veren ayar (özellik
        # boyutu F, PCA boyutu, komşu sayısı K). "test": o TEK ayarın, hiç
        # dokunulmamış test fold'undaki nihai skorları. Ayrımın önemi: test
        # verisiyle ayar seçmek yasaktır; seçim yalnız validasyonla yapılır.
        best = result["best_config"]
        test = result["test"]
        result_rows.append(
            {
                "corpus": corpus,
                "F": best["feature_dim"],
                "PCA": best["pca_dim"],
                "K": best["K"],
                "val_macro_f1": best["val_macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
            }
        )
    show_table(pd.DataFrame(result_rows))

    # Grid aramasının kapsamını kanıtla: toplam kaç kombinasyon denendi ve
    # validasyonda ilk 5 sırayı hangi ayarlar aldı? (En iyinin "tek şanslı
    # atış" olmadığını, sistematik bir taramadan çıktığını gösterir.)
    heading("ODEV 1 - VALIDATION GRID KAPSAMI VE EN IYI 5")
    for corpus in CORPORA:
        path = ODEV1_OUT / corpus / "validation_grid.csv"
        grid = pd.read_csv(path).sort_values("val_macro_f1", ascending=False)
        print(f"\n{corpus}: {len(grid)} kombinasyon | {path.relative_to(ROOT)}")
        show_table(grid.head(5))


def result_params(result: dict) -> str:
    """Modelin hiperparametre sözlüğünü tek satırlık, kararlı bir dizgeye çevirir.

    `sort_keys=True` anahtar sırasını sabitler (her çalıştırmada aynı çıktı),
    `ensure_ascii=True` tabloda hizalamayı bozabilecek karakterleri kaçış
    dizilerine çevirir. Amaç: tabloya sığan, karşılaştırılabilir bir özet.
    """
    return json.dumps(result["best_config"]["params"], sort_keys=True, ensure_ascii=True)


def show_assignment_2() -> None:
    """Ödev 2'nin model karşılaştırmasını, grid kapsamını ve sınıf bazlı skorları gösterir."""
    # Tüm modellerin (KNN dahil) final test karşılaştırması. Sıralama:
    # önce korpusa göre (alfabetik), korpus içinde en yüksek test macro-F1
    # üstte — böylece her korpusun kazananı ilk satırda okunur.
    heading("ODEV 2 - MODEL FINAL TEST SONUCLARI (KNN DAHIL)")
    comparison_path = ODEV2_OUT / "test_comparison_with_knn.csv"
    comparison = pd.read_csv(comparison_path).sort_values(
        ["corpus", "test_macro_f1"], ascending=[True, False]
    )
    # Gösterilecek sütunlar elle seçili: tablo terminale sığsın ve yalnızca
    # karar için önemli alanlar (ayarlar + dört test metriği) öne çıksın.
    columns = [
        "corpus",
        "model",
        "feature_dim",
        "pca_dim",
        "params",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    show_table(comparison[columns])

    # Her model için iki dosya eşleşir: <model>_validation_grid.csv (denenen
    # TÜM kombinasyonlar) ve <model>_result.json (seçilen ayar + test skoru).
    # Dosya adından model anahtarını çıkarıp ikisini birleştiriyoruz; böylece
    # "kaç ayar denendi, hangisi seçildi, test'te ne verdi" tek satırda görünür.
    heading("ODEV 2 - VALIDATION GRID KAPSAMI VE SECILEN AYAR")
    coverage_rows = []
    for corpus in CORPORA:
        for grid_path in sorted((ODEV2_OUT / corpus).glob("*_validation_grid.csv")):
            model_key = grid_path.name.removesuffix("_validation_grid.csv")
            result = read_json(ODEV2_OUT / corpus / f"{model_key}_result.json")
            grid = pd.read_csv(grid_path)
            coverage_rows.append(
                {
                    "corpus": corpus,
                    "model": result["model"],
                    "kombinasyon": len(grid),
                    "feature_dim": result["best_config"]["feature_dim"],
                    "PCA": result["best_config"]["pca_dim"],
                    "parametreler": result_params(result),
                    "val_macro_f1": result["best_config"]["val_macro_f1"],
                    "test_macro_f1": result["test"]["macro_f1"],
                }
            )
    show_table(pd.DataFrame(coverage_rows))

    # Korpus başına en iyi modeli (test macro-F1'e göre) seçip duygu bazında
    # precision/recall/F1 dök: "model hangi duyguları karıştırıyor?" sorusunun
    # cevabı toplam skorda değil, bu sınıf bazlı tabloda görünür.
    heading("ODEV 2 - EN IYI MODELLERIN SINIF BAZLI TEST F1 DEGERLERI")
    per_class_rows = []
    for corpus in CORPORA:
        candidates = []
        for path in (ODEV2_OUT / corpus).glob("*_result.json"):
            result = read_json(path)
            candidates.append(result)
        best = max(candidates, key=lambda item: item["test"]["macro_f1"])
        for emotion, metrics in best["test_per_class"].items():
            per_class_rows.append(
                {
                    "corpus": corpus,
                    "model": best["model"],
                    "emotion": emotion,
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1"], 4),
                    "support": metrics["support"],
                }
            )
    show_table(pd.DataFrame(per_class_rows))

    # PNG görselleri terminalde gösterilemez; onun yerine rapora eklenmeye
    # hazır karmaşıklık (confusion) matrisi dosyalarının yollarını listele.
    heading("KARMASIKLIK MATRISI DOSYALARI")
    for corpus in CORPORA:
        for path in sorted((ODEV2_OUT / corpus).glob("*_confusion_matrix.png")):
            print(path.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını çözümler (--section ile tek bölüm seçilebilir)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=("all", "data", "odev1", "odev2"),
        default="all",
        help="Gosterilecek bolum (varsayilan: all).",
    )
    return parser.parse_args()


def main() -> None:
    """Seçilen bölümleri sırayla yazdırır (varsayılan: hepsi)."""
    args = parse_args()
    if args.section in ("all", "data"):
        show_data_statistics()
    if args.section in ("all", "odev1"):
        show_assignment_1()
    if args.section in ("all", "odev2"):
        show_assignment_2()

    # Kapanış notu: bu betiğin salt-okunur olduğunu ve pytest ile ML test
    # metriklerinin ayrı kavramlar olduğunu okuyucuya hatırlat.
    heading("NOT")
    print("Bu komut mevcut resmi CSV/JSON ciktilarini okur; deney dosyalarini degistirmez.")
    print("pytest kod testidir; ML test metrikleri yukaridaki test fold sonuclaridir.")


if __name__ == "__main__":
    main()
