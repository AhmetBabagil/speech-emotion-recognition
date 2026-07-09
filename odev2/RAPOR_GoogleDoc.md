# Erişim Bilgileri

Ogrenci: Ahmet Babagil - 211101067
Google Drive: https://drive.google.com/drive/folders/1Hbp4WtCGFZjpvQCDxFmqtOmeqq-SMPvW?usp=sharing
GitHub: https://github.com/AhmetBabagil/speech-emotion-recognition/tree/feat/speech-emotion-recognition
Veri seti arsivi: veri_seti_ses.zip teslim klasorune eklendi.

# Project Assignment 2 - Google Doc metni

## 1. Kullanilan kutuphaneler

Bu asamada makine ogrenimi akisi icin yalnizca numpy, pandas ve scikit-learn kullanildi. Grafik cizimi icin matplotlib kullanildi. Ses girdilerini vektore cevirmek icin 1. asamada uretilen donmus Wav2Vec2 oznitelik cache'i kullanildi; bu asamada Wav2Vec2 yeniden egitilmedi ve modelleme tarafinda derin ogrenme kutuphanesi kullanilmadi.

## 2. Veri on isleme ve oznitelik vektorleri

Calismada iki veri seti ayri ayri ele alindi: CREMA-D ve MELD. Iki veri seti de ortak 6 duygu sinifina indirildi: angry, disgust, fear, happy, neutral, sad. Veri bolme islemi konusmaci bagimsiz yapildi; ayni konusmaci train, validation ve test bolumlerinde birlikte bulunmadi.

Her ses kaydi Wav2Vec2 son gizli katmanindan elde edilen havuzlanmis vektorlerle temsil edildi. Oznitelik boyutu hiperparametre olarak ele alindi:

- mean: 768 boyut
- mean + std: 1536 boyut
- mean + std + max: 2304 boyut

Her model icin once bu oznitelik boyutlari dogrudan denendi. Ayrica PCA uygulanarak farkli PCA cikti boyutlari da validasyon seti uzerinde denendi. StandardScaler ve PCA yalnizca egitim verisine uyduruldu; validation/test verisine sadece transform uygulandi.

## 3. Model gelistirme ve model gecerleme protokolu

Her veri seti ve model icin hiperparametre secimi test setine bakmadan, yalnizca validation seti ile yapildi. Secim kriteri once validation macro-F1, esitlik durumunda balanced accuracy ve accuracy olacak sekilde belirlendi. En iyi kombinasyon secildikten sonra model train + validation uzerinde yeniden egitildi ve test seti yalnizca final degerlendirme icin kullanildi.

Denenen modeller ve hiperparametreler:

- Karar Agaci: criterion, max_depth, min_samples_split
- Rastgele Orman: n_estimators, max_depth, max_features
- Gradient Boosting: learning_rate, max_depth

Sinif dengesizligi icin Karar Agaci ve Rastgele Orman modellerinde class_weight="balanced" kullanildi. Gradient Boosting icin sample_weight, balanced class weight mantigiyla verildi.

## 4. Deney kapsami

Deneylerde her veri seti-model ikilisi icin farkli oznitelik boyutlari, PCA secenekleri ve model hiperparametreleri birlikte arandi. Ozet kapsam tablosu `odev2/RAPOR_tablolar.md` dosyasinda verildi. Tam grid dosyalari `odev2/outputs/<veri_seti>/` altinda saklandi.

Ozet olarak:

- Karar Agaci: veri seti basina 216 kombinasyon
- Rastgele Orman: veri seti basina 72 kombinasyon
- Gradient Boosting: veri seti basina 24 kombinasyon

Bu nedenle hiperparametre secimi tek tek elle birkac deger denemek yerine, sistematik validation grid aramasi ile yapildi.

## 5. Test sonuclari ve karsilastirma

Test sonuclari ve KNN dahil genel karsilastirma tablolari `odev2/RAPOR_tablolar.md` dosyasindadir. En onemli bulgular:

- CREMA-D veri setinde en iyi sonuc Gradient Boosting ile elde edildi: test macro-F1 = 0.5144.
- CREMA-D uzerinde Gradient Boosting, 1. asamadaki KNN sonucunu ve Rastgele Orman sonucunu gecti.
- MELD veri setinde en iyi sonuc yine Gradient Boosting ile elde edildi: test macro-F1 = 0.2027.
- MELD sonuclari CREMA-D'ye gore belirgin sekilde dusuktur. Bunun nedeni MELD'in TV diyaloglarindan gelmesi, kayit kosullarinin daha gürultulu/dogal olmasi ve duygu siniflarinin daha zor ayrilmasidir.
- Karar Agaci iki veri setinde de daha dusuk kaldi. Bu durum tek agacin yuksek boyutlu ses ozniteliklerinde genelleme kapasitesinin sinirli kaldigini gosteriyor.
- Rastgele Orman, Karar Agaci'na gore daha iyi sonuc verdi; ensemble yapisi varyansi azaltti. Ancak CREMA-D'de Gradient Boosting daha yuksek macro-F1 elde etti.

## 6. Karmasiklik matrisleri

Her model-veri seti icin test karmasiklik matrisi PNG olarak kaydedildi:

- `odev2/outputs/cremad/decision_tree_confusion_matrix.png`
- `odev2/outputs/cremad/random_forest_confusion_matrix.png`
- `odev2/outputs/cremad/gradient_boosting_confusion_matrix.png`
- `odev2/outputs/meld/decision_tree_confusion_matrix.png`
- `odev2/outputs/meld/random_forest_confusion_matrix.png`
- `odev2/outputs/meld/gradient_boosting_confusion_matrix.png`

Genel olarak CREMA-D'de siniflar MELD'e gore daha ayrilabilir gorundu. MELD'de siniflar arasindaki karismalar daha yuksek oldugu icin macro-F1 daha dusuk kaldi. Bu nedenle yalniz accuracy yerine balanced accuracy ve macro-F1 degerleri de raporlandi.

## 7. Sonuc

Bu asamada iki veri seti icin uc farkli makine ogrenimi modeli gelistirildi, hiperparametreleri validation seti ile secildi ve final performanslari test setiyle olculdu. Sonuclar, ensemble tabanli modellerin tek karar agacina gore daha guclu oldugunu; veri seti kosullarinin da performansi ciddi bicimde etkiledigini gosterdi. CREMA-D kontrollu bir veri seti oldugu icin daha yuksek basari verirken, MELD dogal ve gurultulu konusma ortamindan geldigi icin daha zor bir degerlendirme olusturdu.
