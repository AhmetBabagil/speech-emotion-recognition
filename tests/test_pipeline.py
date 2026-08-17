"""SER işlem hattı için hafif birim testleri (sentetik ses, indirme yok).

Test felsefesi: gerçek veri kümeleri gigabyte'larca yer tutar ve indirmeleri
dakikalar sürer; testler ise saniyeler içinde, her makinede koşabilmelidir.
Bu yüzden burada ses olarak SENTETİK sinüs tonları üretilir ve her duyguya
farklı bir frekans atanır — küçük ama gerçekçi, hatta "ayrıştırılabilir"
bir oyuncak veri kümesi elde edilir.

Çalıştırma:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Depo kökünü sys.path'e ekle: testler, paket pip ile kurulmamış olsa bile
# ``import ser`` yapabilsin (tests/'in bir üst klasörü = depo kökü).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config
from ser.constants import (
    CANONICAL_EMOTIONS, NUM_CLASSES, EMOTION_TO_IDX,
    cremad_code_to_idx, meld_label_to_idx, CORPUS_CREMAD, CORPUS_MELD,
)
from ser.features.io import fix_length
from ser.features.melspec import log_mel_spectrogram, fixed_num_frames, fix_frames
from ser.features.mfcc import mfcc_statistics
from ser.data.splits import prepare_splits


def test_label_maps():
    """Etiket eşlemeleri: iki korpusun etiketleri doğru kanonik indekse gitmeli.

    Bu eşlemelerde sessiz bir kayma, tüm deney sonuçlarını geçersiz kılardı;
    en ucuz sigorta bu testtir.
    """
    assert NUM_CLASSES == 6
    assert cremad_code_to_idx("ANG") == EMOTION_TO_IDX["angry"]
    # Küçük harf de kabul edilmeli (kod .upper() ile normalize ediyor).
    assert cremad_code_to_idx("hap") == EMOTION_TO_IDX["happy"]
    # MELD'in farklı adlandırması bizim kanonik isimlere çevrilmeli.
    assert meld_label_to_idx("joy") == EMOTION_TO_IDX["happy"]
    assert meld_label_to_idx("sadness") == EMOTION_TO_IDX["sad"]
    assert meld_label_to_idx("surprise") is None  # ortak altının dışında -> atılır


def test_config_roundtrip(tmp_path):
    """Config kaydet->yükle turu kayıpsız olmalı (YAML gidiş-dönüşü).

    tmp_path: pytest'in her test için verdiği geçici klasör; test bitince
    otomatik temizlenir, diskte iz kalmaz.
    """
    cfg = Config()
    cfg.feature.fmax = None  # None (opsiyonel değer) durumu da test edilsin
    p = tmp_path / "c.yaml"
    cfg.save(p)
    cfg2 = Config.from_yaml(p)
    assert cfg2.audio.sample_rate == cfg.audio.sample_rate
    assert cfg2.feature.fmax is None
    assert cfg2.model.cnn_channels == cfg.model.cnn_channels
    # YAML tuple'ı liste olarak saklar; yükleyici tuple'a geri çevirmeli.
    assert isinstance(cfg2.model.cnn_channels, tuple)  # list -> tuple dönüşümü


def test_config_ignores_unknown_keys():
    """YAML'daki tanınmayan anahtarlar (üst düzeyde ve alt bölümde) yüklemeyi
    ÇÖKERTMEMELİ — config dosyalarına not düşülebilmesi bilinçli bir esneklik."""
    cfg = Config.from_dict({"model": {"name": "cnn", "bogus": 1}, "totally_unknown": 2})
    assert cfg.model.name == "cnn"


def test_fix_length():
    """Sabit uzunluk yardımcısı: uzunsa kırpmalı, kısaysa doldurmalı."""
    wav = np.ones(1000, dtype=np.float32)
    assert fix_length(wav, 500).shape[0] == 500          # kırpma
    assert fix_length(wav, 2000).shape[0] == 2000        # doldurma


def _tone(freq=220.0, sr=16000, dur=2.0):
    """Sentetik test sinyali: verilen frekansta saf sinüs tonu üretir.

    Gerçek ses dosyası indirmeden öznitelik/model kodunu uçtan uca denemek
    için kullanılır. 0.5 genlik: [-1, 1] aralığında, kırpılma (clipping) yok.
    """
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_shapes():
    """Öznitelik çıkarımının BOYUT sözleşmeleri: modeller bu şekillere güveniyor."""
    cfg = Config()
    wav = _tone()
    mel = log_mel_spectrogram(wav, cfg.feature, cfg.audio.sample_rate)
    assert mel.shape[0] == cfg.feature.n_mels  # satır sayısı = mel bandı sayısı
    # Sabit klibe karşılık gelen kare sayısına kırpınca tam o şekil çıkmalı.
    nf = fixed_num_frames(cfg.audio.num_samples, cfg.feature.hop_length)
    fixed = fix_frames(mel, nf)
    assert fixed.shape == (cfg.feature.n_mels, nf)
    # MFCC istatistik vektörü: n_mfcc * 3 (mfcc, delta, delta2) * 2 (mean, std).
    stats = mfcc_statistics(wav, cfg.feature, cfg.audio.sample_rate)
    assert stats.ndim == 1 and stats.shape[0] == cfg.feature.n_mfcc * 3 * 2


def _synthetic_manifest(tmp_path, n_speakers=6):
    """Küçük sentetik manifest kurar: her (duygu, konuşmacı) çifti için diske
    gerçek bir WAV yazar ve manifest satırı üretir.

    Her duyguya farklı frekans verilir (150 + 50*indeks Hz) — böylece
    kümeleme/sınıflandırma testleri "öğrenilebilir" bir sinyale sahip olur.
    """
    import soundfile as sf
    rows = []
    for e_idx, emo in enumerate(CANONICAL_EMOTIONS):
        for spk in range(n_speakers):
            wav = _tone(freq=150 + 50 * e_idx)
            p = tmp_path / f"spk{spk}_{emo}.wav"
            sf.write(p, wav, 16000)
            rows.append({"path": str(p), "corpus": CORPUS_CREMAD, "speaker": f"spk{spk}",
                         "split": "", "orig_label": emo, "emotion": emo,
                         "label_idx": EMOTION_TO_IDX[emo]})
    return pd.DataFrame(rows)


def test_speaker_independent_split(tmp_path):
    """Konuşmacı-bağımsız bölmenin ana garantisi: foldlar konuşmacı paylaşamaz.

    Ayrıca hiçbir satır kaybolmamalı/çoğalmamalı (toplam korunmalı).
    """
    df = _synthetic_manifest(tmp_path)
    cfg = Config()
    tr, va, te = prepare_splits(df, cfg.data, seed=0)
    # Fold başına konuşmacı kümeleri ikişer ikişer AYRIK olmalı.
    s_tr, s_va, s_te = set(tr.speaker), set(va.speaker), set(te.speaker)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)
    assert len(tr) + len(va) + len(te) == len(df)


def test_cross_corpus_split(tmp_path):
    """Cross-corpus rejimi: eğitim yalnız kaynak korpustan, test yalnız hedef
    korpustan gelmeli (iki alan birbirine karışmamalı)."""
    df = _synthetic_manifest(tmp_path)
    # Aynı manifesti kopyalayıp "meld" korpusuymuş gibi etiketle; konuşmacı
    # adlarına önek ver ki iki korpusun konuşmacıları çakışmasın.
    df2 = df.copy()
    df2["corpus"] = CORPUS_MELD
    df2["speaker"] = "m_" + df2["speaker"]
    full = pd.concat([df, df2], ignore_index=True)
    cfg = Config()
    cfg.data.train_corpora = (CORPUS_CREMAD,)
    cfg.data.eval_corpora = (CORPUS_MELD,)
    tr, va, te = prepare_splits(full, cfg.data, seed=0)
    assert set(tr.corpus) == {CORPUS_CREMAD}
    assert set(te.corpus) == {CORPUS_MELD}


def test_dataset_and_cnn_forward(tmp_path):
    """Uçtan uca duman testi: WAV -> SERDataset -> tensör -> CNN -> logit.

    importorskip: torch kurulu değilse test hata değil "skip" olur —
    torch'suz ortamda da test paketi yeşil kalabilir.
    """
    pytest.importorskip("torch")
    from ser.data.dataset import SERDataset
    from ser.models import build_model

    df = _synthetic_manifest(tmp_path)
    cfg = Config()
    cfg.data.cache_features = False  # test klasörünü .npy önbellekle kirletme
    ds = SERDataset(df, cfg, mode="logmel", train=True)
    x, y = ds[0]
    # Beklenen şekil: [1, n_mels, T] (kanal, frekans, zaman).
    assert x.shape[0] == 1 and x.shape[1] == cfg.feature.n_mels
    model = build_model(cfg, NUM_CLASSES)
    # unsqueeze(0): tek örneğe batch boyutu ekle -> [1, 1, n_mels, T].
    out = model(x.unsqueeze(0))
    assert out.shape == (1, NUM_CLASSES)


def test_augmentation_is_reproducible(tmp_path):
    """Augmentasyon tekrarlanabilirliği: aynı (seed, epoch, index) üçlüsü aynı
    augmentasyonu üretmeli. Bu, "aynı koşuyu tekrar edince aynı sonucu al"
    garantisinin dataset katmanındaki temelidir."""
    torch = pytest.importorskip("torch")
    from ser.data.dataset import SERDataset

    df = _synthetic_manifest(tmp_path)
    cfg = Config()
    cfg.data.cache_features = False
    # Aynı seed + epoch + index -> birebir aynı augmentli örnek (tekrarlanabilir).
    ds1 = SERDataset(df, cfg, mode="logmel", train=True); ds1.set_epoch(3)
    ds2 = SERDataset(df, cfg, mode="logmel", train=True); ds2.set_epoch(3)
    x1, _ = ds1[5]
    x2, _ = ds2[5]
    assert torch.allclose(x1, x2)


def test_class_weights_balanced():
    """Sınıf ağırlıkları: fazla temsil edilen sınıf, nadir sınıftan DÜŞÜK
    ağırlık almalı (dengesizlik telafisinin özü)."""
    pytest.importorskip("torch")
    from ser.data.dataset import class_weights
    # Sınıf 0 aşırı temsil ediliyor -> nadir sınıflardan düşük ağırlık almalı.
    df = pd.DataFrame({"label_idx": [0, 0, 0, 0, 1, 2, 3, 4, 5]})
    w = class_weights(df, "balanced")
    assert w.shape[0] == NUM_CLASSES
    assert w[0] < w[1]


def test_cremad_manifest_parsing(tmp_path):
    """CREMA-D dosya adı ayrıştırma: oyuncu id'si ve duygu, addan doğru çıkmalı."""
    import soundfile as sf
    from ser.data.build_manifest import cremad_rows

    # Gerçek adlandırma kuralına uygun üç sahte WAV üret.
    audiowav = tmp_path / "AudioWAV"
    audiowav.mkdir()
    for n in ["1001_DFA_ANG_XX.wav", "1002_IEO_HAP_HI.wav", "1091_TIE_SAD_LO.wav"]:
        sf.write(audiowav / n, _tone(), 16000)
    rows = cremad_rows(audiowav)
    assert len(rows) == 3
    assert {r["emotion"] for r in rows} == {"angry", "happy", "sad"}
    assert {r["speaker"] for r in rows} == {"1001", "1002", "1091"}  # oyuncu id korunuyor


def test_stratified_subset():
    """Katmanlı alt küme: her sınıftan orantılı ve en az 1 örnek seçilmeli."""
    from ser.semisupervised import _stratified_subset
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)
    rng = np.random.default_rng(0)
    idx = _stratified_subset(y, 0.1, rng)
    # Sınıf başına ~10 örnek; her sınıf temsil edilmeli (>= min_per_class).
    counts = np.bincount(y[idx], minlength=3)
    assert (counts >= 1).all()
    assert 25 <= len(idx) <= 40
    # Sıfıra yakın kesirde bile sınıf başına >= 1 örnek garantisi korunmalı.
    idx2 = _stratified_subset(y, 0.0, rng)
    assert (np.bincount(y[idx2], minlength=3) >= 1).all()


def test_semisupervised_runs(tmp_path):
    """Yarı-denetimli analizler sentetik veride uçtan uca ÇALIŞABİLMELİ.

    Amaç yüksek skor değil, "çökmeden koşuyor ve çıktı sözleşmesine uyuyor"
    (metrikler geçerli aralıkta) garantisi — bir tür bütünleşme duman testi.
    """
    pytest.importorskip("sklearn")
    import soundfile as sf
    from ser.semisupervised import cluster_analysis, label_efficiency

    # Ayrıştırılabilir duygu-başına tonlardan sentetik manifest; birden çok
    # konuşmacı, konuşmacı-bağımsız bölme çalışabilsin diye.
    rows = []
    for e_idx, emo in enumerate(CANONICAL_EMOTIONS):
        for spk in range(8):
            wav = _tone(freq=140 + 60 * e_idx)
            p = tmp_path / f"spk{spk}_{emo}.wav"
            sf.write(p, wav, 16000)
            rows.append({"path": str(p), "corpus": CORPUS_CREMAD, "speaker": f"spk{spk}",
                         "split": "", "orig_label": emo, "emotion": emo,
                         "label_idx": EMOTION_TO_IDX[emo]})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    cfg = Config()
    cfg.data.manifest = str(manifest)
    cfg.data.cache_features = False
    cfg.output_dir = str(tmp_path / "out")

    # Kümeleme: metrikler tanım aralıklarında olmalı (ARI [-1,1], NMI [0,1]).
    clu = cluster_analysis(cfg, tmp_path / "out")
    assert clu["n_clusters"] == NUM_CLASSES
    assert -1.0 <= clu["adjusted_rand_index"] <= 1.0
    assert 0.0 <= clu["normalized_mutual_info"] <= 1.0001

    # Etiket verimliliği: istenen her kesir için bir satır dönmeli.
    eff = label_efficiency(cfg, tmp_path / "out", fractions=(0.5, 1.0))
    assert len(eff) == 2
    assert all(0.0 <= r["macro_f1"] <= 1.0 for r in eff)
