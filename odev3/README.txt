YAP 470 / BİL 570 PROJE İLERLEME RAPORU 3
Konuşmadan Duygu Tanıma - Grup 23

AMAÇ
Bu klasör, CREMA-D ve MELD veri kümeleri üzerinde log-Mel özellikleri ve
sıfırdan geliştirilen PyTorch MLP modelleriyle yapılan üçüncü proje aşamasını
içerir. Model seçimi yalnız validation macro-F1 ile yapılır. Kilitli test
katmanı hiperparametre veya özellik seçmek için kullanılmaz.

HIZLI DOĞRULAMA
1. Tüm testler:
   python -m pytest odev3/tests -q

2. Kayıtlı validation ve test sonuçlarını notebook içinde yeniden hesaplama:
   jupyter nbconvert --to notebook --execute --inplace odev3/validation_test_results.ipynb --ExecutePreprocessor.timeout=120

3. Nihai HTML ve Word raporlarını yeniden üretme:
   python odev3/build_report.py --output-root odev3/outputs --ablation-root odev3/feature_ablation --stability-root odev3/feature_stability --effect-root odev3/hyperparameter_effects

4. Tam model aramasını yeniden çalıştırma:
   python odev3/run_experiment.py --corpora cremad meld --grid-mode report --max-epochs 60 --device cpu --feature-workers 4 --cremad-frame-strategy resize --meld-frame-strategy crop_pad

ANA DOSYALAR

odev3/__init__.py
Python paketini tanımlar.

odev3/run_experiment.py
CREMA-D ve MELD için tarama, yerel iyileştirme, çoklu-seed validation ve
tek seferlik held-out test akışını başlatan komut satırı girişidir.

odev3/pipeline.py
Speaker-independent bölmeleri, resume durumunu, model seçimini, checkpoint,
tahmin, bootstrap belirsizliği ve validation-temelli kalibrasyonu yönetir.

odev3/features_melspec.py
Sesleri 16 kHz mono yükler ve en az 4000 boyutlu log-Mel vektörleri üretir.

odev3/dataset.py
Özellik cache işlemlerini ve yalnız eğitim katında öğrenilen z-score
normalizasyonunu uygular.

odev3/model.py
Hazır ağ kullanmadan yapılandırılabilir PyTorch MLP modelini kurar.

odev3/training.py
Ağırlıklı çapraz entropi, AdamW, gradient clipping, erken durdurma ve
sınıflandırma metriklerini uygular.

odev3/search_space.py
Temel tarama ve kazanan çevresindeki yerel iyileştirme konfigürasyonlarını
üretir.

odev3/feature_ablation.py
Mel bandı, kare boyutu ve zaman ekseni işleme seçeneklerini yalnız validation
katmanında karşılaştırır.

odev3/feature_stability.py
Seçilen ve alternatif Mel temsillerini seed 42, 143 ve 244 ile yeniden eğitip
ortalama, standart sapma ve eşleştirilmiş farkları hesaplar.

odev3/hyperparameter_effects.py
Tarama ve iyileştirme koşularını parametre değerlerine göre gruplayarak
betimsel validation macro-F1 özetlerini üretir.

odev3/uncertainty.py
Held-out test metrikleri için sınıf-korumalı bootstrap güven aralıklarını
hesaplar.

odev3/calibration.py
Validation olasılıklarında temperature scaling katsayısını öğrenir; test NLL,
Brier ve ECE değerlerini ham ve ölçekli olarak hesaplar.

odev3/build_report.py
Kaydedilmiş gerçek deney artefaktlarından HTML ve DOCX raporlarını üretir.

odev3/validation_test_results.ipynb
En iyi hiperparametre satırlarını gösterir, test tahminlerinden accuracy,
balanced accuracy, macro-F1 ve weighted-F1 değerlerini yeniden hesaplar ve
kayıtlı sonuçlarla eşitliğini doğrular. Çalıştırılmış hücre çıktıları dosyada
saklanır.


RAPOR VE SONUÇ KLASÖRLERİ
odev3/outputs/
Her veri kümesi için validation tablosu, seçilmiş model, eğitim geçmişi,
held-out test tahminleri, metrikler, belirsizlik, kalibrasyon ve görseller.


odev3/feature_ablation/
Validation-temelli Mel özellik karşılaştırmalarının CSV, JSON ve PNG çıktıları.


odev3/feature_stability/
İki özellik adayı ve üç seed için ham koşular, özetler, geçmişler ve görseller.


odev3/hyperparameter_effects/
60 benzersiz tarama/iyileştirme konfigürasyonundan üretilen parametre grubu
CSV ve JSON özetleri.


odev3/PROJE_ILERLEME_RAPORU_3.html
Tarayıcıda açılabilen nihai rapor.

odev3/PROJE_ILERLEME_RAPORU_3.docx
Teslim için düzenlenmiş nihai Word raporu.


TEST DOSYALARI

odev3/tests/test_features_melspec.py: Mel özellik boyutu ve sayısal güvenlik.
odev3/tests/test_dataset.py: Cache ve train-only normalizasyon davranışı.
odev3/tests/test_model.py: MLP katmanları, boyutlar ve ileri geçiş.
odev3/tests/test_training.py: Eğitim, metrik ve erken durdurma davranışı.
odev3/tests/test_search_space.py: Tarama ve iyileştirme konfigürasyonları.
odev3/tests/test_pipeline.py: Bölme, resume, seçim ve held-out test koruması.
odev3/tests/test_cli.py: Komut satırı argümanları.
odev3/tests/test_feature_ablation.py: Özellik karşılaştırma protokolü.
odev3/tests/test_feature_stability.py: Çoklu-seed özellik doğrulaması.
odev3/tests/test_hyperparameter_effects.py: Gruplu parametre etki analizi.
odev3/tests/test_uncertainty.py: Sınıf-korumalı bootstrap hesapları.
odev3/tests/test_calibration.py: Olasılık kalibrasyonu ve karar koruması.
odev3/tests/test_report.py: HTML/DOCX içerik ve artefakt bütünlüğü.
odev3/tests/test_submission_requirements.py: VS Code ayarı, README ve çalıştırılmış notebook teslim koşulları.


TEKRARLANABİLİRLİK NOTLARI

Rastgele seed değerleri çıktı dosyalarında saklanır. Veri normalizasyonu ve
sınıf ağırlıkları yalnız eğitim katından öğrenilir. Model/özellik seçimi yalnız
validation sonuçlarına dayanır. Test tahminleri seçim tamamlandıktan sonra bir
kez üretilir. Raporlanan bütün sayılar odev3 altındaki CSV ve JSON dosyalarından
yeniden oluşturulabilir.
