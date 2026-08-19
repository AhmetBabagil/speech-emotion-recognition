# Deney defteri: tüm deney klasörlerindeki Yöntem 2 (RNN) test sonuçlarını tek listede toplar.
import json  # summary.json dosyalarını okumak için
from pathlib import Path  # dosya yolları

KOK = Path("final/deneyler")  # deney çıktılarının kök klasörü
print("koşu | test doğruluk | test macro-F1")  # tablo başlığı


for dosya in sorted(KOK.glob("*/cremad/summary.json")):  # her deneyin özet dosyası için
    veri = json.loads(dosya.read_text(encoding="utf-8"))  # özeti oku
    ad = dosya.parts[2]  # yolun 3. parcasi (deney adı)
    print(ad, "|", round(veri["rnn"]["test"]["accuracy"], 4), "|", round(veri["rnn"]["test"]["macro_f1"], 4))  # sonucu yazdır
