# VS Code Local History Durumu

Kontrol edilen klasor:
C:\Users\ahmet\AppData\Roaming\Code\User\History

Sonuc:
- VS Code Local History klasoru sistemde mevcut.
- Ancak `odev2` dosyalari icin Local History kaydi bulunamadi.
- Bunun nedeni bu dosyalarin VS Code yerine terminal/komut satiri tarafindan olusturulmasidir.

Yapilmasi gereken manuel adim:
1. VS Code ile `odev2` klasorunu ac.
2. `PROJE_ILERLEME_RAPORU_2.md`, `SONUCLAR.ipynb`, `model_pipeline.py`, `run_experiment.py`, `build_report.py` dosyalarini ac.
3. Her dosyada kucuk bir yorum/bosluk degisikligi yapip kaydet.
4. Explorer > Timeline bolumunde Local History kaydi olustugunu kontrol et.
5. Hocanin istedigi sekilde VS Code Local History kayitlarini zipleyip Drive'a ekle.

Not:
Proje kodu, rapor, notebook ve deney ciktilari `submission/odev2_project.zip` icinde hazirdir. Bu zip VS Code Local History zipinin yerine gecmez; Local History ayrica VS Code tarafindan alinmalidir.
