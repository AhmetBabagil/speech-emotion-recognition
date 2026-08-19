# Modelden bağımsız, ağırlıklı loss + geçerleme temelli early stopping'li eğitim.
#
# Aynı döngü hem Yöntem 1'in CNN'ini hem Yöntem 2'nin LSTM/GRU'sunu eğitir: tek varsayımı, modelin bir öznitelik yığınını sınıf logit'lerine çevirmesidir. Model seçimi geçerleme macro-F1 üzerinden yapılır ve eğitim bittiğinde en iyi epoch'un ağırlıkları geri yüklenir.
#
# Yönergenin iki açık şartı bu dosyada karşılanır:
# 1. "Sınıf veri sayıları farklıysa hata fonksiyonunu veri sayısıyla ters orantılı ağırlıklı tanımlayın" -> inverse_frequency_weights
# 2. "PyTorch ile erken durdurma yapın" -> train_with_early_stopping

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from dataclasses import dataclass  # sonuç sınıfını (TrainingOutcome) kolay yazmak için
from typing import Any, Callable  # tip ipuçları: "herhangi bir tip" ve "fonksiyon"

import numpy as np  # sayısal diziler
import torch  # PyTorch
from torch import nn  # sinir ağı katmanları + kayıp fonksiyonları
from torch.utils.data import DataLoader  # veriyi yığınlar hâlinde besleyen yükleyici

from final.dataset import ArrayDataset  # numpy dizilerini torch veri kümesine saran sınıf
from final.models import OptimSettings  # eğitim ayarları (lr, batch, patience...)
from ser.evaluate import compute_metrics  # doğruluk, macro-F1 vb. hesaplayan yardımcı
from ser.utils import set_seed  # tüm rastgeleliği sabitleyen tohum ayarı


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    # Sınıf ağırlıkları: n / (K * n_k) — eğitimdeki sınıf sayısıyla ters orantılı.
    #
    # Örnek: bir sınıfın az kaydı varsa ağırlığı büyük olur; böylece loss, çoğunluk sınıfını ezberleyip azınlığı görmezden gelmeyi cezalandırır.

    labels = np.asarray(labels, dtype=np.int64)  # etiketleri tam sayı dizisine çevir
    counts = np.bincount(labels, minlength=num_classes)  # her sınıfta kaç örnek var say
    if len(labels) == 0 or len(counts) != num_classes or np.any(counts == 0):  # her sınıf temsil edilmeli
        raise ValueError(
            f'Her sınıf eğitim verisinde bulunmalı; sayımlar={counts.tolist()}.'  # yoksa hata
        )
    weights = len(labels) / (num_classes * counts.astype(np.float64))  # n / (K * n_k) formülü
    return torch.tensor(weights, dtype=torch.float32)  # ağırlıkları torch tensörü olarak döndür


def make_loader(
    features: np.ndarray,  # öznitelikler
    labels: np.ndarray,  # etiketler
    *,
    batch_size: int,  # yığın boyu
    shuffle: bool,  # sıra karışsın mı
    seed: int,  # tekrarlanabilirlik tohumu
    device: torch.device,  # cpu/cuda
    num_workers: int = 0,  # veri yükleme işçisi sayısı
    drop_last: bool = False,  # son eksik yığını düşür mü
) -> DataLoader:  # Tekrarlanabilir (tohumlu) bir DataLoader kurar.

    # Karıştırma sırası bu üretece bağlı; sabit tohum = aynı sıra = aynı sonuç.
    generator = torch.Generator()  # karıştırma için rastgele üreteç
    generator.manual_seed(seed)  # üreteci sabit tohumla başlat (tekrarlanabilir)
    return DataLoader(  # yığın besleyen yükleyiciyi kur
        ArrayDataset(features, labels),  # numpy -> torch veri kümesi
        batch_size=batch_size,  # yığın boyu
        shuffle=shuffle,  # karıştır mı
        drop_last=drop_last,  # son eksik yığını at mı
        num_workers=num_workers,  # işçi sayısı
        pin_memory=device.type == 'cuda',   # GPU'ya kopyalamayı hızlandırır
        persistent_workers=num_workers > 0,  # işçileri epoch'lar arası canlı tut
        generator=generator,  # karıştırma üreteci
    )


def evaluate_loader(
    model: nn.Module,  # değerlendirilecek model
    loader: DataLoader,  # veri yükleyici
    criterion: nn.Module,  # kayıp fonksiyonu
    device: torch.device,  # cpu/cuda
) -> tuple[float, dict[str, Any], np.ndarray]:  # Modeli gradyansız değerlendirir: (ortalama loss, metrikler, olasılıklar).

    model.eval()   # BatchNorm/Dropout'u değerlendirme kipine al
    total_loss = 0.0  # toplam kayıp
    total_examples = 0  # toplam örnek sayısı
    true_parts: list[np.ndarray] = []  # gerçek etiketleri biriktir
    predicted_parts: list[np.ndarray] = []  # tahminleri biriktir
    probability_parts: list[np.ndarray] = []  # olasılıkları biriktir

    with torch.no_grad():   # değerlendirmede geri yayılım yok -> bellek/hız kazancı
        for features, labels in loader:  # her yığın için
            features = features.to(device, non_blocking=True)  # öznitelikleri cihaza taşı
            labels = labels.to(device, non_blocking=True)  # etiketleri cihaza taşı
            logits = model(features)  # modelden ham puanlar (logit)
            loss = criterion(logits, labels)  # kaybı hesapla
            probabilities = torch.softmax(logits, dim=1)  # puanları olasılığa çevir
            batch = labels.shape[0]  # bu yığındaki örnek sayısı
            total_loss += float(loss.item()) * batch  # kaybı örnek sayısıyla ağırlıkla topla
            total_examples += batch  # toplam örneği güncelle
            true_parts.append(labels.cpu().numpy())  # gerçek etiketleri sakla
            predicted_parts.append(probabilities.argmax(dim=1).cpu().numpy())  # en yüksek olasılık = tahmin
            probability_parts.append(probabilities.cpu().numpy())  # olasılıkları sakla

    if total_examples == 0:  # hiç örnek yoksa
        raise ValueError('Değerlendirme yükleyicisi hiç örnek üretmedi.')  # hata
    y_true = np.concatenate(true_parts)  # tüm gerçek etiketleri birleştir
    y_pred = np.concatenate(predicted_parts)  # tüm tahminleri birleştir
    return (
        total_loss / total_examples,  # ortalama kayıp
        compute_metrics(y_true, y_pred),  # metrikler (doğruluk, macro-F1...)
        np.concatenate(probability_parts),  # tüm olasılıklar
    )


@dataclass
class TrainingOutcome:  # Bir eğitim koşusunun tüm sonucu: model + geçmiş + en iyi epoch bilgileri.

    model: nn.Module  # eğitilmiş (en iyi epoch'a yüklenmiş) model
    history: list[dict[str, Any]]           # epoch başına metrikler (öğrenme eğrisi)
    best_epoch: int                         # geçerlemede en iyi olan epoch
    epochs_trained: int                     # fiilen koşulan epoch sayısı
    stopped_early: bool                     # erken mi durdu?
    validation_loss: float                  # en iyi epoch'taki geçerleme loss'u
    validation_metrics: dict[str, Any]      # en iyi epoch'taki geçerleme metrikleri


def train_with_early_stopping(
    model: nn.Module,  # eğitilecek model (CNN ya da RNN)
    train_features: np.ndarray,  # eğitim öznitelikleri
    train_labels: np.ndarray,  # eğitim etiketleri
    validation_features: np.ndarray,  # geçerleme öznitelikleri
    validation_labels: np.ndarray,  # geçerleme etiketleri
    optim: OptimSettings,  # eğitim ayarları
    *,
    num_classes: int,  # sınıf sayısı (6)
    device: torch.device,  # cpu/cuda
    max_epochs: int = 60,  # en fazla epoch
    seed: int = 42,  # tohum
    num_workers: int = 0,  # veri işçisi
    amp: bool = True,  # karışık hassasiyet (GPU hızlandırma)
    min_delta: float = 1e-4,  # "anlamlı iyileşme" eşiği
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,  # eğitim artırması (opsiyonel)
    label_smoothing: float = 0.0,  # etiket yumuşatma (0 = kapalı)
    mixup_alpha: float = 0.0,  # mixup gücü (0 = kapalı)
) -> TrainingOutcome:
    # Bir adayı eğitir ve geçerleme macro-F1'i en iyi olan epoch'u geri yükler.
    #
    # Early stopping mantığı: her epoch sonunda geçerleme macro-F1 ölçülür; `patience` epoch boyunca anlamlı iyileşme (min_delta'dan büyük) olmazsa eğitim durur. Böylece model, geçerlemede bozulmaya başladığı (aşırı öğrenme) noktadan sonrasını "unutur".
    #
    # ``train_transform`` (ör. SpecAugment maskeleme) SADECE eğitim yığınlarına uygulanır; geçerleme dokunulmadan kalır ki model seçimi yansız (tarafsız) olsun.

    optim.validate()  # eğitim ayarlarını doğrula
    if max_epochs <= 0:  # epoch sayısı pozitif olmalı
        raise ValueError(f'max_epochs pozitif olmalı, gelen {max_epochs}.')  # değilse hata
    set_seed(seed)   # python/numpy/torch tohumları -> tekrarlanabilirlik

    model = model.to(device)  # modeli cihaza (GPU) taşı
    # Yönerge şartı: sınıf sayısıyla ters orantılı ağırlıklı cross-entropy.
    class_weights = inverse_frequency_weights(train_labels, num_classes).to(device)  # sınıf ağırlıkları
    # label_smoothing>0: modelin "kesin eminim" demesini cezalandırır; hedef
    # olasılık kütlesinin küçük bir kısmı diğer sınıflara dağıtılır (varsayılan 0 = kapalı).
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)  # ağırlıklı kayıp
    optimizer = torch.optim.AdamW(  # AdamW eniyileyici
        model.parameters(),  # güncellenecek ağırlıklar
        lr=optim.learning_rate,  # öğrenme oranı
        weight_decay=optim.weight_decay,  # L2 düzenlileştirme
    )

    # BatchNorm istatistik hesaplayabilmek için her eğitim yığınında en az
    # 2 örnek ister; son yığın tek örnekli kalacaksa onu düşür.
    drop_last = (
        len(train_labels) > optim.batch_size  # birden fazla yığın varsa
        and len(train_labels) % optim.batch_size == 1  # ve son yığın tek örnekli kalacaksa
    )
    train_loader = make_loader(  # eğitim yükleyicisi
        train_features,
        train_labels,
        batch_size=optim.batch_size,
        shuffle=True,            # eğitimde sıra her epoch karışır
        seed=seed,
        device=device,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    validation_loader = make_loader(  # geçerleme yükleyicisi
        validation_features,
        validation_labels,
        batch_size=max(optim.batch_size, 128),  # değerlendirmede büyük yığın (hızlı)
        shuffle=False,           # değerlendirmede karıştırmaya gerek yok
        seed=seed,
        device=device,
        num_workers=num_workers,
    )

    # AMP (karışık hassasiyet): GPU'da float16 ile daha hızlı eğitim;
    # GradScaler sayısal taşmaları dengeler. CPU'da otomatik kapalı.
    use_amp = bool(amp and device.type == 'cuda')  # AMP yalnız GPU'da açık
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # float16 taşmalarını dengeleyen ölçekleyici
    history: list[dict[str, Any]] = []  # öğrenme eğrisi kaydı
    best_state: dict[str, torch.Tensor] | None = None  # en iyi epoch'un ağırlıkları
    best_epoch = 0  # en iyi epoch numarası
    best_score = -float('inf')  # şimdiye kadarki en iyi macro-F1
    best_loss = float('inf')  # en iyi epoch'un loss'u
    best_metrics: dict[str, Any] | None = None  # en iyi epoch'un metrikleri
    epochs_without_improvement = 0  # kaç epoch iyileşme olmadı (sabır sayacı)

    for epoch in range(1, max_epochs + 1):  # her epoch için (1'den başla)
        # ---- 1) Eğitim geçişi -------------------------------------------------
        model.train()  # modeli eğitim kipine al (Dropout/BatchNorm aktif)
        running_loss = 0.0  # bu epoch'un toplam kaybı
        examples_seen = 0  # bu epoch'ta görülen örnek sayısı
        train_true: list[np.ndarray] = []  # eğitim gerçek etiketleri
        train_pred: list[np.ndarray] = []  # eğitim tahminleri
        for features, labels in train_loader:  # her eğitim yığını için
            features = features.to(device, non_blocking=True)  # öznitelikleri cihaza taşı
            labels = labels.to(device, non_blocking=True)  # etiketleri cihaza taşı
            if train_transform is not None:
                # Veri artırma (varsa) yalnızca burada, eğitim yığınında.
                features = train_transform(features)  # SpecAugment vb. uygula
            # Mixup (Zhang vd., 2018): iki örneği (ve etiketlerini) rastgele lam
            # oranıyla karıştır; karar sınırını yumuşatan güçlü bir düzenlileştirme.
            # mixup_alpha=0 -> tamamen kapalı (mevcut davranış birebir korunur).
            mix_labels = None  # mixup kapalıysa None
            mix_lam = 1.0  # karışım oranı (1 = karışım yok)
            if mixup_alpha > 0.0:  # mixup açıksa
                mix_lam = float(np.random.beta(mixup_alpha, mixup_alpha))  # Beta dağılımından oran çek
                perm = torch.randperm(features.shape[0], device=features.device)  # yığını rastgele karıştır
                features = mix_lam * features + (1.0 - mix_lam) * features[perm]  # iki örneği karıştır
                mix_labels = labels[perm]  # karışan örneğin etiketi
            optimizer.zero_grad(set_to_none=True)  # önceki gradyanları temizle
            with torch.amp.autocast('cuda', enabled=use_amp):  # AMP kapsamı (float16)
                logits = model(features)  # ileri geçiş: puanlar
                if mix_labels is None:  # mixup yoksa
                    loss = criterion(logits, labels)  # normal kayıp
                else:  # mixup varsa
                    loss = (mix_lam * criterion(logits, labels)  # iki etiketin ağırlıklı kaybı
                            + (1.0 - mix_lam) * criterion(logits, mix_labels))
            scaler.scale(loss).backward()      # geri yayılım
            scaler.unscale_(optimizer)  # gradyanları ölçekten geri al (kırpmadan önce)
            # Gradyan kırpma: tek bir kötü yığının eğitimi raydan çıkarmasını önler.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # gradyan büyüklüğünü sınırla
            scaler.step(optimizer)             # ağırlık güncelle
            scaler.update()  # ölçekleyiciyi güncelle
            batch = labels.shape[0]  # yığın boyu
            running_loss += float(loss.item()) * batch  # kaybı topla
            examples_seen += batch  # örnek sayısını güncelle
            train_true.append(labels.detach().cpu().numpy())  # gerçek etiketleri sakla
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())  # tahminleri sakla

        # ---- 2) Geçerleme geçişi ---------------------------------------------
        train_loss = running_loss / max(examples_seen, 1)  # epoch ortalama eğitim kaybı
        train_metrics = compute_metrics(np.concatenate(train_true), np.concatenate(train_pred))  # eğitim metrikleri
        val_loss, val_metrics, _ = evaluate_loader(model, validation_loader, criterion, device)  # geçerlemede ölç

        # ---- 3) En iyiyi güncelle + erken durdurma sayacı --------------------
        score = float(val_metrics['macro_f1'])  # model seçimi ölçütü: geçerleme macro-F1
        improved = score > best_score + min_delta  # anlamlı bir iyileşme oldu mu
        # Deterministik beraberlik bozma: macro-F1 sayısal olarak eşitse
        # geçerleme loss'u düşük olan kazanır; ama minicik değişimler
        # sabır sayacını sıfırlamaz (yoksa eğitim gereksiz uzar).
        tied_but_lower_loss = abs(score - best_score) <= min_delta and val_loss < best_loss  # eşitlikte loss'a bak
        if improved or tied_but_lower_loss:  # daha iyi (ya da eşit+düşük loss) ise
            best_score = score  # en iyi skoru güncelle
            best_loss = val_loss  # en iyi loss'u güncelle
            best_epoch = epoch  # en iyi epoch'u kaydet
            best_metrics = val_metrics  # en iyi metrikleri kaydet
            # En iyi anın ağırlıklarının kopyasını CPU'da sakla.
            best_state = {
                name: value.detach().cpu().clone()  # her ağırlığın kopyası
                for name, value in model.state_dict().items()
            }
        if improved:  # gerçek iyileşme varsa
            epochs_without_improvement = 0  # sabır sayacını sıfırla
        else:  # iyileşme yoksa
            epochs_without_improvement += 1  # sabır sayacını artır

        # Öğrenme eğrisi için epoch kaydı.
        history.append(  # bu epoch'un metriklerini geçmişe ekle
            {
                'epoch': epoch,  # epoch numarası
                'train_loss': train_loss,  # eğitim kaybı
                'train_accuracy': train_metrics['accuracy'],  # eğitim doğruluğu
                'train_macro_f1': train_metrics['macro_f1'],  # eğitim macro-F1
                'val_loss': val_loss,  # geçerleme kaybı
                'val_accuracy': val_metrics['accuracy'],  # geçerleme doğruluğu
                'val_balanced_accuracy': val_metrics['balanced_accuracy'],  # dengeli doğruluk
                'val_macro_f1': val_metrics['macro_f1'],  # geçerleme macro-F1 (seçim ölçütü)
                'val_weighted_f1': val_metrics['weighted_f1'],  # ağırlıklı F1
                'improved': bool(improved or tied_but_lower_loss),  # bu epoch en iyiyi güncelledi mi
            }
        )
        # ERKEN DURDURMA: sabır dolduysa döngüden çık.
        if epochs_without_improvement >= optim.patience:  # patience kadar iyileşme yoksa
            break  # eğitimi durdur

    if best_state is None or best_metrics is None:  # hiç geçerli kontrol noktası yoksa
        raise RuntimeError('Eğitim, geçerli bir geçerleme kontrol noktası üretmeden bitti.')  # hata
    # En iyi epoch'un ağırlıklarını geri yükle: dönen model, eğitimin
    # sonundaki değil geçerlemede EN İYİ olan modeldir.
    model.load_state_dict(best_state)  # en iyi ağırlıkları modele yükle
    return TrainingOutcome(  # sonucu paketleyip döndür
        model=model,  # en iyi model
        history=history,  # öğrenme eğrisi
        best_epoch=best_epoch,  # en iyi epoch
        epochs_trained=len(history),  # koşulan epoch sayısı
        stopped_early=len(history) < max_epochs,  # erken mi durdu
        validation_loss=best_loss,  # en iyi loss
        validation_metrics=best_metrics,  # en iyi metrikler
    )


def evaluate_arrays(
    model: nn.Module,  # değerlendirilecek model
    features: np.ndarray,  # öznitelikler
    labels: np.ndarray,  # etiketler
    *,
    class_weights: torch.Tensor,  # sınıf ağırlıkları (kayıp için)
    device: torch.device,  # cpu/cuda
    batch_size: int = 256,  # yığın boyu
    num_workers: int = 0,  # işçi sayısı
) -> tuple[float, dict[str, Any], np.ndarray]:  # Sabit bir modeli karıştırmadan ve gradyansız değerlendirir (test için).

    loader = make_loader(  # test yükleyicisi (karıştırmasız)
        features,
        labels,
        batch_size=batch_size,
        shuffle=False,  # test sırası sabit
        seed=0,
        device=device,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # ağırlıklı kayıp
    return evaluate_loader(model, loader, criterion, device)  # değerlendir ve sonucu döndür
