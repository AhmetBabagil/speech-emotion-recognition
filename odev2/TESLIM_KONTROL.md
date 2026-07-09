# Odev 2 teslim kontrol listesi

## Google Doc'a eklenecek icerik

- Ana dosya: `odev2/PROJE_ILERLEME_RAPORU_2.md`
- Word rapor: `odev2/PROJE_ILERLEME_RAPORU_2.docx`
- PDF rapor: `odev2/PROJE_ILERLEME_RAPORU_2.pdf`
- Alternatif birlesik rapor: `odev2/RAPOR_TAM.md`
- Bu dosya, anlatim metni + validation/test tablolarini birlikte icerir.
- Sadece tablo istenirse: `odev2/RAPOR_tablolar.md`
- Daha kisa anlatim metni istenirse: `odev2/RAPOR_GoogleDoc.md`

## Submission kopyalari

- `submission/Ahmet_Babagil_211101067_Rapor2.docx`
- `submission/Ahmet_Babagil_211101067_Rapor2.pdf`
- `submission/odev2_project.zip`

## Kod ve cikti dosyalari

- Notebook sonuc dosyasi: `odev2/SONUCLAR.ipynb`
- Deney kodu: `odev2/model_pipeline.py`
- Calistirma CLI: `odev2/run_experiment.py`
- Rapor uretici: `odev2/build_report.py`
- Word uretici: `odev2/build_docx.py`
- Ozet karsilastirma: `odev2/outputs/test_comparison_with_knn.csv`
- Model bazli test karsilastirma: `odev2/outputs/model_comparison.csv`
- Validation gridleri: `odev2/outputs/<veri_seti>/*_validation_grid.csv`
- Test sonuclari: `odev2/outputs/<veri_seti>/*_result.json`
- Karmasiklik matrisleri: `odev2/outputs/<veri_seti>/*_confusion_matrix.png`

## Deney kapsami

- CREMA-D + MELD ayri ayri calistirildi.
- Karar Agaci: veri seti basina 216 validation kombinasyonu.
- Rastgele Orman: veri seti basina 72 validation kombinasyonu.
- Gradient Boosting: veri seti basina 24 validation kombinasyonu.
- Secim validation macro-F1'a gore yapildi; test seti sadece final degerlendirme icin kullanildi.

## VS Code ayari

- `odev2/.vscode/settings.json` mevcut ve hocanin istedigi ayarlari iceriyor.
- Kok klasorde de `.vscode/settings.json` mevcut.
- VS Code'da proje klasoru olarak `odev2` klasoru acilabilir.
- Timeline / Local History kontrol edilmeli.

## Local History uyarisi

- Local History kontrol notu: `odev2/LOCAL_HISTORY_DURUMU.md`

VS Code Local History, VS Code'un kendi User/History klasorunde tutulur. Bu dosyalar otomatik olarak proje klasorune gelmez. Teslimden once VS Code Timeline bolumunden `odev2` icin degisen dosyalarin local history kayitlari gorulmeli ve dersin istedigi sekilde ziplenmelidir.

## Hazirlanan Drive teslim klasoru

- submission 2/ klasoru Drive'a yuklemek icin hazirlanacaktir.
- Icinde rapor DOCX/PDF, odev2_project.zip, eri_seti_ses.zip, kod/cikti kopyalari ve teslim notlari bulunacaktir.
