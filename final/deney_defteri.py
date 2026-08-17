import json 
from pathlib import Path

KOK = Path("final/deneyler")
print("koşu | test doğruluk | test macro-F1")


for dosya in sorted(KOK.glob("*/cremad/summary.json")):
    veri = json.loads(dosya.read_text(encoding="utf-8"))
    ad = dosya.parts[2]  # yolun 3. parcasi
    print(ad, "|", round(veri["rnn"]["test"]["accuracy"], 4), "|", round(veri["rnn"]["test"]["macro_f1"], 4))
