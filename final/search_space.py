# İki final yöntemi için deterministik hiperparametre adayları.
#
# Her aday, bir "öznitelik ayarı + model ayarı" İKİLİSİDİR. Bu sayede öznitelik düzeyindeki hiperparametreler (Yöntem 1'de mel çözünürlüğü; Yöntem 2'de ödevin şart koştuğu ARALIK SAYISI ve GENİŞLİĞİ) geçerleme kümesinde model hiperparametreleriyle tamamen aynı şekilde aranır.
#
# Listeler bilerek elle yazılmıştır (rastgele örnekleme yok): böylece aynı komut her koşuda aynı adayları dener ve deney tekrarlanabilir olur.
#
# Üç mod vardır:
# - quick : dakikalar içinde biten minik duman testi (2'şer aday)
# - report: rapor için ana mod — güçlü bir taban + tek-faktör değişimleri (her seferde TEK ayar değişir, böylece hangi ayarın ne etki ettiği tablodan doğrudan okunur)
# - full  : daha geniş kartezyen tarama (zaman varsa)

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from dataclasses import replace  # bir ayarın tek alanını değiştirip kopyasını üretmek için

from final.features import IntervalConfig, MelImageConfig  # öznitelik ayarları
from final.models import CNNConfig, OptimSettings, RNNConfig  # model + optimizasyon ayarları

GRID_MODES = ('quick', 'report', 'full')  # geçerli arama modları

# Tip takma adları: (öznitelik ayarı, model ayarı) ikilisi.
CNNCandidate = tuple[MelImageConfig, CNNConfig]  # CNN adayı türü
RNNCandidate = tuple[IntervalConfig, RNNConfig]  # RNN adayı türü


def _unique(candidates):  # Adayları doğrular ve yinelenenleri sırayı bozmadan eler.

    result, seen = [], set()  # sonuç listesi + görülenler kümesi
    for feature_cfg, model_cfg in candidates:  # her aday için
        feature_cfg.validate()  # öznitelik ayarını doğrula
        model_cfg.validate()  # model ayarını doğrula
        key = (feature_cfg, model_cfg)   # frozen dataclass'lar hash'lenebilir
        if key not in seen:  # daha önce görülmediyse
            seen.add(key)  # görülenlere ekle
            result.append(key)  # sonuca ekle
    return result  # benzersiz adaylar


# --- Yöntem 1: mel görüntüsü + CNN -----------------------------------------------

CNN_BASE_FEATURES = MelImageConfig()   # 64 mel x 128 kare
CNN_BASE_MODEL = CNNConfig()           # 32-64-128 kanal, dropout 0.3


def cnn_space(mode: str) -> list[CNNCandidate]:  # Yöntem 1 için aday listesi.

    if mode == 'quick':  # duman testi
        # Duman testi: küçük görüntü + küçük ağ, 2 epoch'ta biter.
        small = MelImageConfig(n_mels=32, n_frames=48)  # küçük mel görüntüsü
        return _unique([
            (small, CNNConfig(channels=(8, 16), dropout=0.0,  # minik ağ
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
            (small, CNNConfig(channels=(16, 32), dropout=0.3,  # biraz daha büyük
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
        ])
    if mode == 'report':  # rapor modu (ana)
        base_f, base_m = CNN_BASE_FEATURES, CNN_BASE_MODEL  # taban öznitelik + model
        base_o = base_m.optim  # taban optimizasyon ayarı
        return _unique([
            (base_f, base_m),                                        # taban aday
            # --- tek-faktör model/optimizasyon değişimleri ---
            (base_f, replace(base_m, channels=(16, 32, 64))),         # daha küçük ağ
            (base_f, replace(base_m, channels=(32, 64, 128, 256))),   # daha derin ağ
            (base_f, replace(base_m, dropout=0.1)),                   # az düzenlileştirme
            (base_f, replace(base_m, dropout=0.5)),                   # çok düzenlileştirme
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=1e-3))),  # yüksek lr
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=1e-4))),  # düşük lr
            (base_f, replace(base_m, optim=replace(base_o, batch_size=64))),  # büyük yığın
            (base_f, replace(base_m, optim=replace(base_o, weight_decay=0.0))),  # düzenlileştirme yok
            # --- tek-faktör öznitelik değişimleri ---
            (replace(base_f, n_frames=96), base_m),                   # daha kısa zaman ekseni
            (replace(base_f, n_mels=80), base_m),                     # daha ince frekans çözünürlüğü
        ])
    if mode == 'full':  # geniş tarama
        # Kartezyen tarama: kanal x dropout x öğrenme oranı.
        candidates = []  # aday biriktir
        for channels in ((16, 32, 64), (32, 64, 128), (32, 64, 128, 256)):  # kanal seçenekleri
            for dropout in (0.1, 0.3, 0.5):  # dropout seçenekleri
                for lr in (1e-3, 3e-4):  # öğrenme oranı seçenekleri
                    candidates.append((  # her kombinasyonu ekle
                        CNN_BASE_FEATURES,
                        CNNConfig(channels=channels, dropout=dropout,
                                  optim=replace(CNN_BASE_MODEL.optim, learning_rate=lr)),
                    ))
        return _unique(candidates)  # benzersizleri döndür
    raise ValueError(f'Bilinmeyen mod {mode!r}; beklenen: {GRID_MODES}.')  # tanımsız mod -> hata


def cnn_refinement(winner: CNNCandidate, exclude=()) -> list[CNNCandidate]:
    # Geçerleme kazananının çevresinde ikinci tur yerel arama.
    #
    # Mantık: geniş aramanın kazananı iyi bir bölgeyi işaret eder; o bölgede her ayarı bir tık aşağı/yukarı oynatıp daha da iyisi var mı bakılır.

    feature_cfg, model_cfg = winner  # kazananın öznitelik + model ayarı
    optim = model_cfg.optim  # kazananın optimizasyon ayarı
    scaled_down = tuple(max(8, c // 2) for c in model_cfg.channels)    # yarı genişlik
    scaled_up = tuple(min(512, c * 2) for c in model_cfg.channels)     # çift genişlik
    candidates = _unique([
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=optim.learning_rate / 2))),  # lr yarı
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=min(3e-3, optim.learning_rate * 2)))),  # lr çift
        (feature_cfg, replace(model_cfg, optim=replace(optim, batch_size=max(16, optim.batch_size // 2)))),  # yığın yarı
        (feature_cfg, replace(model_cfg, optim=replace(optim, batch_size=min(128, optim.batch_size * 2)))),  # yığın çift
        (feature_cfg, replace(model_cfg, dropout=round(max(0.0, model_cfg.dropout - 0.1), 2))),  # dropout az
        (feature_cfg, replace(model_cfg, dropout=round(min(0.6, model_cfg.dropout + 0.1), 2))),  # dropout çok
        (feature_cfg, replace(model_cfg, channels=scaled_down)),  # ağ yarı genişlik
        (feature_cfg, replace(model_cfg, channels=scaled_up)),  # ağ çift genişlik
    ])
    # Kazananın kendisi ve daha önce denenmiş adaylar tekrar denenmez.
    excluded = set(exclude) | {winner}  # elenecek adaylar
    return [c for c in candidates if c not in excluded]  # yeni adaylar


# --- Yöntem 2: aralık serisi + LSTM/GRU ------------------------------------------

RNN_BASE_FEATURES = IntervalConfig()   # 24 aralık x 300 ms
RNN_BASE_MODEL = RNNConfig()           # BiGRU 192 x 2 katman, mean pooling


def rnn_space(mode: str) -> list[RNNCandidate]:  # Yöntem 2 için aday listesi.

    if mode == 'quick':  # duman testi
        small = IntervalConfig(n_intervals=8, interval_ms=200)  # az aralık
        return _unique([
            (small, RNNConfig(rnn_type='gru', hidden_size=32, num_layers=1, bidirectional=False,  # minik GRU
                              dropout=0.0, pooling='mean',
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
            (small, RNNConfig(rnn_type='lstm', hidden_size=32, num_layers=1, bidirectional=True,  # minik LSTM
                              dropout=0.0, pooling='last',
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
        ])
    if mode == 'report':  # rapor modu (ana)
        base_f, base_m = RNN_BASE_FEATURES, RNN_BASE_MODEL  # taban öznitelik + model
        base_o = base_m.optim  # taban optimizasyon
        return _unique([
            (base_f, base_m),                                        # taban aday
            # --- ARALIK DÜZENİ birinci sınıf hiperparametre (ödev şartı) ---
            (replace(base_f, n_intervals=16), base_m),                # daha az aralık
            (replace(base_f, n_intervals=32), base_m),                # daha çok aralık
            (replace(base_f, interval_ms=200), base_m),               # daha dar pencere
            (replace(base_f, interval_ms=400), base_m),               # daha geniş pencere
            # --- tek-faktör model/optimizasyon değişimleri ---
            (base_f, replace(base_m, rnn_type='lstm')),               # GRU yerine LSTM
            (base_f, replace(base_m, hidden_size=96)),                # daha küçük gizli durum
            (base_f, replace(base_m, hidden_size=256)),               # daha büyük gizli durum
            (base_f, replace(base_m, num_layers=1)),                  # tek katman
            (base_f, replace(base_m, bidirectional=False)),           # tek yönlü
            (base_f, replace(base_m, pooling='last')),                # son adım özeti
            (base_f, replace(base_m, pooling='max')),                 # maksimum özeti
            (base_f, replace(base_m, dropout=0.1)),                   # az dropout
            (base_f, replace(base_m, dropout=0.5)),                   # çok dropout
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=3e-4))),  # düşük lr
            (base_f, replace(base_m, optim=replace(base_o, batch_size=32))),  # küçük yığın
        ])
    if mode == 'full':  # geniş tarama
        # Aralık düzeni x model tipi x gizli boyut kartezyeni.
        candidates = []  # aday biriktir
        for n_intervals in (16, 24, 32):  # aralık sayısı seçenekleri
            for interval_ms in (200, 300, 400):  # aralık genişliği seçenekleri
                feature_cfg = IntervalConfig(n_intervals=n_intervals, interval_ms=interval_ms)  # öznitelik ayarı
                for rnn_type in ('gru', 'lstm'):  # model tipi seçenekleri
                    for hidden in (96, 192):  # gizli boyut seçenekleri
                        candidates.append((  # kombinasyonu ekle
                            feature_cfg,
                            replace(RNN_BASE_MODEL, rnn_type=rnn_type, hidden_size=hidden),
                        ))
        return _unique(candidates)  # benzersizleri döndür
    raise ValueError(f'Bilinmeyen mod {mode!r}; beklenen: {GRID_MODES}.')  # tanımsız mod -> hata


def rnn_refinement(winner: RNNCandidate, exclude=()) -> list[RNNCandidate]:
    # Geçerleme kazananının çevresinde ikinci tur yerel arama.
    #
    # Not: aralık sayısı/genişliği de burada oynatılır — nihai kazanan (32 aralık x 200 ms) tam olarak bu turda bulunmuştur.

    feature_cfg, model_cfg = winner  # kazananın ayarları
    optim = model_cfg.optim  # optimizasyon ayarı
    other_type = 'lstm' if model_cfg.rnn_type == 'gru' else 'gru'  # diğer RNN tipi
    candidates = _unique([
        # Aralık düzenini bir adım aşağı/yukarı oynat.
        (replace(feature_cfg, n_intervals=max(8, feature_cfg.n_intervals - 8)), model_cfg),  # daha az aralık
        (replace(feature_cfg, n_intervals=min(48, feature_cfg.n_intervals + 8)), model_cfg),  # daha çok aralık
        (replace(feature_cfg, interval_ms=max(100, feature_cfg.interval_ms - 100)), model_cfg),  # daha dar
        (replace(feature_cfg, interval_ms=min(600, feature_cfg.interval_ms + 100)), model_cfg),  # daha geniş
        # Model tarafında yerel değişimler.
        (feature_cfg, replace(model_cfg, rnn_type=other_type)),  # tipi değiştir
        (feature_cfg, replace(model_cfg, hidden_size=max(32, model_cfg.hidden_size // 2))),  # gizli yarı
        (feature_cfg, replace(model_cfg, hidden_size=min(512, model_cfg.hidden_size * 2))),  # gizli çift
        (feature_cfg, replace(model_cfg, dropout=round(max(0.0, model_cfg.dropout - 0.1), 2))),  # dropout az
        (feature_cfg, replace(model_cfg, dropout=round(min(0.6, model_cfg.dropout + 0.1), 2))),  # dropout çok
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=optim.learning_rate / 2))),  # lr yarı
    ])
    excluded = set(exclude) | {winner}  # elenecekler
    return [c for c in candidates if c not in excluded]  # yeni adaylar
