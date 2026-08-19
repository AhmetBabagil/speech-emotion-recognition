# Derin modeller için eğitim döngüsü (log-mel üzerinde CNN, dalga formu üzerinde wav2vec2).
#
# Bu modül modern bir eğitim döngüsünün "iyi uygulamalarını" bir arada içerir:
# * Sınıf-ağırlıklı kayıp (dengesiz veri için) + etiket yumuşatma,
# * Kosinüs öğrenme hızı çizelgesi (lr'yi yumuşakça sıfıra indirir),
# * AMP — otomatik karışık hassasiyet (GPU'da float16 ile hız/bellek kazancı),
# * Gradyan kırpma (patlayan gradyanlara karşı sigorta),
# * Doğrulama metriğine göre erken durdurma (overfitting başlayınca durur),
# * En iyi epoch'un checkpoint'ini kaydetme ve test için geri yükleme,
# * Test kümesi raporu (metrics.json + confusion matrix PNG).
# Dönüş değeri: test metrikleri sözlüğü.

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import NUM_CLASSES
from .config import Config
from .data import SERDataset, class_weights, prepare_splits
from .evaluate import evaluate_torch, report, compute_metrics
from .models import build_model
from .utils import get_logger, set_seed, get_device, ensure_dir, count_params

log = get_logger(__name__)


def _make_loaders(cfg, train_df, val_df, test_df, device):
    # Üç fold için DataLoader'ları kurar (train karıştırılır, val/test asla).
    #
    # Modelin türüne göre veri temsili seçilir: wav2vec2 ham dalga formu ister, CNN log-mel spektrogram ister — ikisi de aynı SERDataset sınıfından ``mode`` parametresiyle elde edilir.
    import torch
    from torch.utils.data import DataLoader

    mode = "waveform" if cfg.model.name.lower() == "wav2vec2" else "logmel"
    # pin_memory yalnızca CUDA'da işe yarar: sabitlenmiş (page-locked) RAM'den
    # GPU'ya kopyalama daha hızlıdır; CPU'da gereksiz maliyettir.
    pin = device.type == "cuda"
    common = dict(num_workers=cfg.train.num_workers, pin_memory=pin)
    # Tohumlanmış generator -> shuffle sırası da tekrarlanabilir olur,
    # num_workers > 0 olsa bile. (Aksi hâlde her koşuda batch sırası değişirdi.)
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        SERDataset(train_df, cfg, mode=mode, train=True),
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=False,
        generator=g, **common,
    )
    # val/test: shuffle=False ve train=False -> deterministik değerlendirme
    # (merkez kırpma, augmentasyon yok); skorlar koşudan koşuya oynamaz.
    val_loader = DataLoader(
        SERDataset(val_df, cfg, mode=mode, train=False),
        batch_size=cfg.train.batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        SERDataset(test_df, cfg, mode=mode, train=False),
        batch_size=cfg.train.batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, test_loader


def _amp_tools(cfg, device):
    # AMP (karışık hassasiyet) araçlarını hazırlar: (scaler, autocast fabrikası).
    #
    # AMP fikri: ileri/geri geçişin çoğunu float16 ile yapmak (hızlı, az bellek), fakat float16'nın dar aralığında küçük gradyanlar sıfıra yuvarlanabildiği için kaybı önce büyütüp (scale) sonra geri küçültmek — GradScaler tam bunu yapar. AMP kapalıyken aynı nesneler "hiçbir şey yapmadan" çalışır; böylece eğitim döngüsü tek bir kod yoluyla hem CPU hem GPU'da koşar.
    import torch

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    try:
        # Yeni torch API'si: torch.amp.GradScaler("cuda", ...)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:  # eski torch sürümleri için geri-uyumluluk
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def autocast():
        # AMP açıksa gerçek autocast bağlamı, kapalıysa "boş" bağlam döndür.
        # nullcontext sayesinde çağıran taraf if/else yazmak zorunda kalmaz.
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    return scaler, autocast


def _train_one_epoch(model, loader, criterion, optimizer, scaler, autocast, device, grad_clip):  # Tek epoch eğitim: tüm batch'leri gezer, ortalama kaybı döndürür.
    import torch

    model.train()  # Dropout/BatchNorm'u eğitim moduna al
    total, running = 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        # set_to_none=True: gradyanları sıfır tensörle doldurmak yerine None
        # yapar — bir tık daha hızlı ve bellek dostu.
        optimizer.zero_grad(set_to_none=True)
        with autocast():  # (AMP açıksa) ileri geçiş float16'da
            logits = model(xb)
            loss = criterion(logits, yb)
        # scale(loss): float16'da gradyan taşmasını önlemek için kaybı büyüt.
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            # Kırpmadan önce unscale şart: aksi hâlde eşik, büyütülmüş
            # gradyanlarla karşılaştırılır ve kırpma yanlış ölçekte yapılırdı.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)   # (gerekirse) unscale edip optimizer adımını at
        scaler.update()          # bir sonraki adım için ölçeği ayarla
        # Kayıp ortalaması batch büyüklüğüyle ağırlıklandırılır: son batch
        # küçük olsa bile epoch ortalaması doğru çıkar.
        bs = yb.size(0)
        running += loss.item() * bs
        total += bs
    return running / max(total, 1)


def train_torch(cfg: Config, device=None) -> dict:  # Uçtan uca eğitim: veri böl, modeli kur, eğit, en iyiyi test et, raporla.
    import torch
    import torch.nn as nn

    set_seed(cfg.train.seed)   # tekrarlanabilirlik: her şeyden önce tohumla
    device = device or get_device()
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    # Kullanılan config'i çıktıya kaydet: deney kaydı/tekrarı için.
    cfg.save(out_dir / "config.yaml")

    df = pd.read_csv(cfg.data.manifest)
    # prepare_splits() hiçbir foldun boş kalmadığını zaten garanti eder
    # (boşsa açıklayıcı bir hata fırlatır).
    train_df, val_df, test_df = prepare_splits(df, cfg.data, cfg.train.seed)

    train_loader, val_loader, test_loader = _make_loaders(cfg, train_df, val_df, test_df, device)

    model = build_model(cfg, NUM_CLASSES).to(device)
    log.info("Model '%s' with %s trainable params", cfg.model.name, f"{count_params(model):,}")

    # Sınıf ağırlıkları TRAIN dağılımından hesaplanır (val/test'e bakılmaz) ve
    # kayba verilir: nadir sınıfların hatası daha pahalı olur.
    weights = class_weights(train_df, cfg.train.class_weighting).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.train.label_smoothing)
    # AdamW: L2 cezasını gradyandan ayrıştıran (decoupled) Adam türevi.
    # Yalnızca requires_grad=True parametreler verilir: wav2vec2'nin dondurulmuş
    # katmanları optimizer'ın defterinde bile yer almasın.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    # Kosinüs çizelge: lr, epochs boyunca kosinüs eğrisiyle 0'a iner —
    # başta büyük adımlarla keşif, sonda küçük adımlarla ince ayar.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler, autocast = _amp_tools(cfg, device)

    # En iyi skor -inf'ten başlar: ilk epoch her durumda "en iyi" sayılıp kaydedilir.
    best_score, best_epoch, best_path = -np.inf, -1, out_dir / "best.pt"
    history = []
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        # Her epoch farklı augmentasyon görsün ama aynı koşu tekrarında yine
        # aynısı üretilsin (epoch numarası RNG tohumuna karışır).
        train_loader.dataset.set_epoch(epoch)
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, autocast, device, cfg.train.grad_clip
        )
        # Her epoch sonunda doğrulama: izlenen metrik (varsayılan makro-F1)
        # erken durdurma ve "en iyi model" seçiminin pusulasıdır.
        y_true, y_pred, _ = evaluate_torch(model, val_loader, device)
        vmetrics = compute_metrics(y_true, y_pred)
        score = vmetrics[cfg.train.monitor]
        scheduler.step()  # lr'yi çizelgeye göre güncelle (epoch başına bir kez)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_accuracy": vmetrics["accuracy"], "val_macro_f1": vmetrics["macro_f1"]})
        log.info("epoch %3d | loss %.4f | val_acc %.4f | val_macroF1 %.4f%s",
                 epoch, train_loss, vmetrics["accuracy"], vmetrics["macro_f1"],
                 "  *" if score > best_score else "")

        if score > best_score:
            # Yeni rekor: sayaçları sıfırla ve checkpoint'i üzerine yaz.
            best_score, best_epoch, epochs_no_improve = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": cfg.to_dict()}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.early_stop_patience:
                # Sabır doldu: model artık doğrulamada iyileşmiyor; devam etmek
                # yalnızca overfitting'i derinleştirir.
                log.info("Early stopping at epoch %d (best epoch %d, %s=%.4f)",
                         epoch, best_epoch, cfg.train.monitor, best_score)
                break

    # Test, SON epoch'un değil EN İYİ epoch'un ağırlıklarıyla yapılır — erken
    # durdurmanın bütün amacı budur.
    if best_path.exists():
        # weights_only=False: checkpoint'imiz ağırlıkların yanında config
        # sözlüğünü de taşıyor (weights_only=True yalnızca tensör yükleyebilirdi).
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        log.info("Loaded best checkpoint from epoch %d", state["epoch"])

    y_true, y_pred, _ = evaluate_torch(model, test_loader, device)
    metrics = report(y_true, y_pred, out_dir, prefix="test",
                     title=f"{cfg.experiment} (test)")
    metrics["best_epoch"] = best_epoch
    metrics["val_best_" + cfg.train.monitor] = best_score
    # Eğitim eğrisi (epoch başına loss/val skorları) ayrı dosyaya: sonradan
    # öğrenme eğrisi çizmek için ham veri.
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    # Kısa özet: koca confusion matrix'i hariç tutarak okunaklı bir JSON.
    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    return metrics


def train_baseline(cfg: Config, kind: str = "svm") -> dict:
    # Klasik MFCC-istatistik taban modelini (sklearn) eğitir ve değerlendirir.
    #
    # Derin modellerle aynı manifest, aynı bölme ve aynı raporlama kullanılır; tek fark öğrenicinin sklearn pipeline'ı olmasıdır. Böylece "derin model klasik yönteme göre ne kazandırıyor?" sorusu adil şekilde yanıtlanır.
    from .data import mfcc_feature_matrix
    from .models import build_baseline

    set_seed(cfg.train.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    cfg.save(out_dir / "config.yaml")

    df = pd.read_csv(cfg.data.manifest)
    # prepare_splits() hiçbir foldun boş kalmadığını zaten garanti eder.
    train_df, val_df, test_df = prepare_splits(df, cfg.data, cfg.train.seed)
    # Yalnızca TRAIN foldu ile fit edilir (val dahil edilmez): derin modeller de
    # aynı veriden öğrendiği için karşılaştırma adil ve sızıntısız olur.
    # Pipeline içindeki StandardScaler da böylece yalnızca train istatistikleriyle
    # fit edilmiş olur (test bilgisi ölçekleme üzerinden bile sızmaz).
    fit_df = train_df

    log.info("Extracting MFCC-statistics features ...")
    X_train, y_train = mfcc_feature_matrix(fit_df, cfg)
    X_test, y_test = mfcc_feature_matrix(test_df, cfg)

    pipe = build_baseline(kind)
    log.info("Fitting baseline '%s' on %d samples (dim=%d) ...", kind, len(y_train), X_train.shape[1])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = report(y_test, y_pred, out_dir, prefix="test",
                     title=f"{cfg.experiment} baseline-{kind} (test)")
    try:
        # Eğitilen pipeline'ı diske kaydet (sonradan tahmin için yeniden
        # eğitmeye gerek kalmasın). joblib, numpy dizileri için pickle'dan verimlidir.
        import joblib
        joblib.dump(pipe, out_dir / "baseline.joblib")
    except Exception as e:
        # Kaydetme başarısız olsa bile metrikler elimizde; koşuyu düşürme.
        log.warning("Could not save baseline model: %s", e)
    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    return metrics
