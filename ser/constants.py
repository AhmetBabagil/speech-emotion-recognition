"""Kanonik (standart) etiket uzayı ve korpusa özgü etiket eşlemeleri.

Bu modül, projenin tamamında kullanılan duygu sınıfları için TEK doğruluk
kaynağıdır (single source of truth). Neden tek bir yerde? Çünkü etiket
listesi iki farklı dosyada tanımlanırsa, bir gün biri güncellenir diğeri
unutulur ve sınıf indeksleri sessizce kayar — bu tür bir hata modeli
"çalışır ama yanlış öğrenir" hâle getirir ve fark etmesi çok zordur.

Her iki korpus da AYNI altı kanonik etikete eşlenir; böylece korpus-içi ve
korpuslar-arası değerlendirme tek bir ortak indeks uzayını paylaşır.

Ortak altı duygu (CREMA-D ile MELD'in kesişimi):
    angry, disgust, fear, happy, neutral, sad

MELD'de ek olarak ``surprise`` (şaşkınlık) vardır; CREMA-D'de bu sınıf
bulunmadığı için ``surprise`` → ``None`` eşlenir ve bu kayıtlar birleşik
manifest'ten tamamen çıkarılır. Aksi hâlde iki korpusun sınıf kümeleri
uyuşmaz ve çapraz değerlendirme adil olmazdı.
"""

from __future__ import annotations

# --- Kanonik etiket uzayı -------------------------------------------------------
# Sıralı ve SABİT bir liste: bir duygunun bu listedeki indeksi, onun sınıf
# kimliğidir (class id). Alfabetik sırada tutulur ki sıralama keyfî olmasın;
# modelin çıktı katmanındaki 0..5 nöronları her zaman aynı duyguya karşılık gelir.
CANONICAL_EMOTIONS: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# İki yönlü arama sözlükleri: isim → indeks ve indeks → isim.
# Eğitim sırasında etiketleri sayıya, raporlama sırasında sayıları isme çevirmek
# için ikisine de ihtiyaç duyulur.
EMOTION_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}
IDX_TO_EMOTION: dict[int, str] = {i: e for e, i in EMOTION_TO_IDX.items()}
NUM_CLASSES: int = len(CANONICAL_EMOTIONS)

# --- CREMA-D --------------------------------------------------------------------
# CREMA-D'de etiket, dosya adının İÇİNDE kodludur. Dosya adı deseni:
#   <ActorID>_<Sentence>_<Emotion>_<Level>.wav   örn. 1001_DFA_ANG_XX.wav
# Alt çizgiyle ayrılan ÜÇÜNCÜ parça duygu kodudur (ANG, DIS, ...).
# Bu sözlük, 3 harfli kodları kanonik isimlere çevirir.
CREMAD_CODE_TO_CANONICAL: dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# --- MELD -----------------------------------------------------------------------
# MELD'de etiketler CSV dosyalarında metin olarak durur: *_sent_emo.csv
# dosyalarının "Emotion" sütunu bu küçük harfli dizgeleri kullanır.
# Dikkat: MELD'in isimlendirmesi bizimkiyle birebir aynı değildir
# ("anger" ≠ "angry", "joy" ≠ "happy", "sadness" ≠ "sad") — bu sözlük tam da
# bu çeviriyi yapar. ``surprise`` ortak altılının dışında olduğundan → None
# (yani o satırlar atılır).
MELD_LABEL_TO_CANONICAL: dict[str, str | None] = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": None,
}

# --- Ses (audio) varsayılanları -------------------------------------------------
# Yükleme/yeniden örnekleme sonrası TÜM sesler için hedef örnekleme frekansı.
# 16 kHz konuşma işlemede standarttır: insan konuşmasının ayırt edici bilgisi
# büyük ölçüde 8 kHz altındadır (Nyquist sınırı = 16000/2) ve wav2vec2 gibi
# önceden eğitilmiş modeller de girişlerini 16 kHz'de bekler.
SAMPLE_RATE: int = 16_000

# Manifest dosyasının "corpus" sütununda kullanılan korpus kimlikleri.
# Sabit olarak tanımlanır ki kodun her yerinde "cremad" yazarken yapılacak bir
# yazım hatası çalışma zamanında değil, import sırasında yakalanabilsin.
CORPUS_CREMAD = "cremad"
CORPUS_MELD = "meld"


def cremad_code_to_idx(code: str) -> int | None:
    """CREMA-D'nin 3 harfli duygu kodunu kanonik sınıf indeksine çevirir.

    Kod tanınmıyorsa None döner (çağıran taraf o kaydı atlar). ``.upper()``
    sayesinde "ang" gibi küçük harfli girişler de tolere edilir — dosya
    adlarının büyük/küçük harf tutarlılığına güvenmek zorunda kalmayız.
    """
    canon = CREMAD_CODE_TO_CANONICAL.get(code.upper())
    return EMOTION_TO_IDX[canon] if canon is not None else None


def meld_label_to_idx(label: str) -> int | None:
    """MELD duygu dizgesini kanonik sınıf indeksine çevirir.

    Ortak altılıda olmayan etiketler (örn. "surprise") için None döner.
    ``.strip().lower()`` ile CSV'den gelebilecek baştaki/sondaki boşluklar ve
    büyük harf farklılıkları normalize edilir; ham veriye asla körü körüne
    güvenilmez.
    """
    canon = MELD_LABEL_TO_CANONICAL.get(label.strip().lower())
    return EMOTION_TO_IDX[canon] if canon is not None else None
