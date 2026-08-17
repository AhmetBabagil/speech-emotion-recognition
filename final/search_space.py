'''İki final yöntemi için deterministik hiperparametre adayları.

Her aday, bir "öznitelik ayarı + model ayarı" İKİLİSİDİR. Bu sayede
öznitelik düzeyindeki hiperparametreler (Yöntem 1'de mel çözünürlüğü;
Yöntem 2'de ödevin şart koştuğu ARALIK SAYISI ve GENİŞLİĞİ) geçerleme
kümesinde model hiperparametreleriyle tamamen aynı şekilde aranır.

Listeler bilerek elle yazılmıştır (rastgele örnekleme yok): böylece aynı
komut her koşuda aynı adayları dener ve deney tekrarlanabilir olur.

Üç mod vardır:
- quick : dakikalar içinde biten minik duman testi (2'şer aday)
- report: rapor için ana mod — güçlü bir taban + tek-faktör değişimleri
  (her seferde TEK ayar değişir, böylece hangi ayarın ne etki ettiği
  tablodan doğrudan okunur)
- full  : daha geniş kartezyen tarama (zaman varsa)
'''

from __future__ import annotations

from dataclasses import replace

from final.features import IntervalConfig, MelImageConfig
from final.models import CNNConfig, OptimSettings, RNNConfig

GRID_MODES = ('quick', 'report', 'full')

# Tip takma adları: (öznitelik ayarı, model ayarı) ikilisi.
CNNCandidate = tuple[MelImageConfig, CNNConfig]
RNNCandidate = tuple[IntervalConfig, RNNConfig]


def _unique(candidates):
    '''Adayları doğrular ve yinelenenleri sırayı bozmadan eler.'''

    result, seen = [], set()
    for feature_cfg, model_cfg in candidates:
        feature_cfg.validate()
        model_cfg.validate()
        key = (feature_cfg, model_cfg)   # frozen dataclass'lar hash'lenebilir
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


# --- Yöntem 1: mel görüntüsü + CNN -----------------------------------------------

CNN_BASE_FEATURES = MelImageConfig()   # 64 mel x 128 kare
CNN_BASE_MODEL = CNNConfig()           # 32-64-128 kanal, dropout 0.3


def cnn_space(mode: str) -> list[CNNCandidate]:
    '''Yöntem 1 için aday listesi.'''

    if mode == 'quick':
        # Duman testi: küçük görüntü + küçük ağ, 2 epoch'ta biter.
        small = MelImageConfig(n_mels=32, n_frames=48)
        return _unique([
            (small, CNNConfig(channels=(8, 16), dropout=0.0,
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
            (small, CNNConfig(channels=(16, 32), dropout=0.3,
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
        ])
    if mode == 'report':
        base_f, base_m = CNN_BASE_FEATURES, CNN_BASE_MODEL
        base_o = base_m.optim
        return _unique([
            (base_f, base_m),                                        # taban aday
            # --- tek-faktör model/optimizasyon değişimleri ---
            (base_f, replace(base_m, channels=(16, 32, 64))),         # daha küçük ağ
            (base_f, replace(base_m, channels=(32, 64, 128, 256))),   # daha derin ağ
            (base_f, replace(base_m, dropout=0.1)),                   # az düzenlileştirme
            (base_f, replace(base_m, dropout=0.5)),                   # çok düzenlileştirme
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=1e-3))),
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=1e-4))),
            (base_f, replace(base_m, optim=replace(base_o, batch_size=64))),
            (base_f, replace(base_m, optim=replace(base_o, weight_decay=0.0))),
            # --- tek-faktör öznitelik değişimleri ---
            (replace(base_f, n_frames=96), base_m),                   # daha kısa zaman ekseni
            (replace(base_f, n_mels=80), base_m),                     # daha ince frekans çözünürlüğü
        ])
    if mode == 'full':
        # Kartezyen tarama: kanal x dropout x öğrenme oranı.
        candidates = []
        for channels in ((16, 32, 64), (32, 64, 128), (32, 64, 128, 256)):
            for dropout in (0.1, 0.3, 0.5):
                for lr in (1e-3, 3e-4):
                    candidates.append((
                        CNN_BASE_FEATURES,
                        CNNConfig(channels=channels, dropout=dropout,
                                  optim=replace(CNN_BASE_MODEL.optim, learning_rate=lr)),
                    ))
        return _unique(candidates)
    raise ValueError(f'Bilinmeyen mod {mode!r}; beklenen: {GRID_MODES}.')


def cnn_refinement(winner: CNNCandidate, exclude=()) -> list[CNNCandidate]:
    '''Geçerleme kazananının çevresinde ikinci tur yerel arama.

    Mantık: geniş aramanın kazananı iyi bir bölgeyi işaret eder; o bölgede
    her ayarı bir tık aşağı/yukarı oynatıp daha da iyisi var mı bakılır.
    '''

    feature_cfg, model_cfg = winner
    optim = model_cfg.optim
    scaled_down = tuple(max(8, c // 2) for c in model_cfg.channels)    # yarı genişlik
    scaled_up = tuple(min(512, c * 2) for c in model_cfg.channels)     # çift genişlik
    candidates = _unique([
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=optim.learning_rate / 2))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=min(3e-3, optim.learning_rate * 2)))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, batch_size=max(16, optim.batch_size // 2)))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, batch_size=min(128, optim.batch_size * 2)))),
        (feature_cfg, replace(model_cfg, dropout=round(max(0.0, model_cfg.dropout - 0.1), 2))),
        (feature_cfg, replace(model_cfg, dropout=round(min(0.6, model_cfg.dropout + 0.1), 2))),
        (feature_cfg, replace(model_cfg, channels=scaled_down)),
        (feature_cfg, replace(model_cfg, channels=scaled_up)),
    ])
    # Kazananın kendisi ve daha önce denenmiş adaylar tekrar denenmez.
    excluded = set(exclude) | {winner}
    return [c for c in candidates if c not in excluded]


# --- Yöntem 2: aralık serisi + LSTM/GRU ------------------------------------------

RNN_BASE_FEATURES = IntervalConfig()   # 24 aralık x 300 ms
RNN_BASE_MODEL = RNNConfig()           # BiGRU 192 x 2 katman, mean pooling


def rnn_space(mode: str) -> list[RNNCandidate]:
    '''Yöntem 2 için aday listesi.'''

    if mode == 'quick':
        small = IntervalConfig(n_intervals=8, interval_ms=200)
        return _unique([
            (small, RNNConfig(rnn_type='gru', hidden_size=32, num_layers=1, bidirectional=False,
                              dropout=0.0, pooling='mean',
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
            (small, RNNConfig(rnn_type='lstm', hidden_size=32, num_layers=1, bidirectional=True,
                              dropout=0.0, pooling='last',
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
        ])
    if mode == 'report':
        base_f, base_m = RNN_BASE_FEATURES, RNN_BASE_MODEL
        base_o = base_m.optim
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
            (base_f, replace(base_m, dropout=0.1)),
            (base_f, replace(base_m, dropout=0.5)),
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=3e-4))),
            (base_f, replace(base_m, optim=replace(base_o, batch_size=32))),
        ])
    if mode == 'full':
        # Aralık düzeni x model tipi x gizli boyut kartezyeni.
        candidates = []
        for n_intervals in (16, 24, 32):
            for interval_ms in (200, 300, 400):
                feature_cfg = IntervalConfig(n_intervals=n_intervals, interval_ms=interval_ms)
                for rnn_type in ('gru', 'lstm'):
                    for hidden in (96, 192):
                        candidates.append((
                            feature_cfg,
                            replace(RNN_BASE_MODEL, rnn_type=rnn_type, hidden_size=hidden),
                        ))
        return _unique(candidates)
    raise ValueError(f'Bilinmeyen mod {mode!r}; beklenen: {GRID_MODES}.')


def rnn_refinement(winner: RNNCandidate, exclude=()) -> list[RNNCandidate]:
    '''Geçerleme kazananının çevresinde ikinci tur yerel arama.

    Not: aralık sayısı/genişliği de burada oynatılır — nihai kazanan
    (32 aralık x 200 ms) tam olarak bu turda bulunmuştur.
    '''

    feature_cfg, model_cfg = winner
    optim = model_cfg.optim
    other_type = 'lstm' if model_cfg.rnn_type == 'gru' else 'gru'
    candidates = _unique([
        # Aralık düzenini bir adım aşağı/yukarı oynat.
        (replace(feature_cfg, n_intervals=max(8, feature_cfg.n_intervals - 8)), model_cfg),
        (replace(feature_cfg, n_intervals=min(48, feature_cfg.n_intervals + 8)), model_cfg),
        (replace(feature_cfg, interval_ms=max(100, feature_cfg.interval_ms - 100)), model_cfg),
        (replace(feature_cfg, interval_ms=min(600, feature_cfg.interval_ms + 100)), model_cfg),
        # Model tarafında yerel değişimler.
        (feature_cfg, replace(model_cfg, rnn_type=other_type)),
        (feature_cfg, replace(model_cfg, hidden_size=max(32, model_cfg.hidden_size // 2))),
        (feature_cfg, replace(model_cfg, hidden_size=min(512, model_cfg.hidden_size * 2))),
        (feature_cfg, replace(model_cfg, dropout=round(max(0.0, model_cfg.dropout - 0.1), 2))),
        (feature_cfg, replace(model_cfg, dropout=round(min(0.6, model_cfg.dropout + 0.1), 2))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=optim.learning_rate / 2))),
    ])
    excluded = set(exclude) | {winner}
    return [c for c in candidates if c not in excluded]
