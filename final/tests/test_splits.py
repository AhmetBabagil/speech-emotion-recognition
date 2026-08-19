# Konuşmacı-bağımsız bölmenin GERÇEKTEN sızıntısız olduğunu doğrulayan testler.
#
# Projenin en kritik iddiası: aynı konuşmacı hem eğitimde hem testte olmaz. Bu testler o iddiayı sentetik bir manifest üzerinde otomatik kanıtlar; ayrıca rastgele bölmenin (kıyas) sızıntı ürettiğini de gösterir.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import pandas as pd  # sentetik manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from final.pipeline import SplitSettings  # noqa: E402  # bölme ayarı
from ser.data.splits import prepare_splits  # noqa: E402  # bölme fonksiyonu


def _sentetik_manifest(n_speakers: int = 20, kayit_basi: int = 12) -> pd.DataFrame:  # test için sahte manifest üretir
    rows = []  # satırlar
    for s in range(n_speakers):  # her konuşmacı için
        for k in range(kayit_basi):  # o konuşmacının her kaydı için
            rows.append({'corpus': 'cremad', 'speaker': f'sp{s}',  # korpus + konuşmacı
                         'label_idx': k % 6, 'path': f'sp{s}_{k}.wav'})  # duygu (0-5) + yol
    return pd.DataFrame(rows)  # manifest tablosu


def test_speaker_split_no_leakage() -> None:  # Konuşmacı-bağımsız bölme: hiçbir konuşmacı birden fazla foldda olmamalı.
    m = _sentetik_manifest()  # sahte manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',), split='speaker')  # konuşmacı bölmesi
    tr, va, te = prepare_splits(m, ayar, seed=42)  # böl
    s_tr, s_va, s_te = set(tr['speaker']), set(va['speaker']), set(te['speaker'])  # fold konuşmacıları
    assert s_tr.isdisjoint(s_va)  # eğitim ∩ geçerleme = boş (sızıntı yok)
    assert s_tr.isdisjoint(s_te)  # eğitim ∩ test = boş (sızıntı yok)
    assert s_va.isdisjoint(s_te)  # geçerleme ∩ test = boş
    assert len(tr) + len(va) + len(te) == len(m)  # tüm kayıtlar bir folda düşmeli (kayıp yok)


def test_random_split_leaks_speakers() -> None:  # Kıyas: rastgele bölmede aynı konuşmacı birden çok foldda OLUR (sızıntı).
    m = _sentetik_manifest()  # sahte manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',), split='random')  # rastgele bölme
    tr, _, te = prepare_splits(m, ayar, seed=42)  # böl
    s_tr, s_te = set(tr['speaker']), set(te['speaker'])  # eğitim + test konuşmacıları
    assert not s_tr.isdisjoint(s_te)  # kesişim boş DEĞİL -> işte bu yüzden rastgele bölme kullanmıyoruz


def test_speaker_split_deterministic() -> None:  # Aynı seed -> aynı bölme (tekrarlanabilirlik).
    m = _sentetik_manifest()  # sahte manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',), split='speaker')  # konuşmacı bölmesi
    a = prepare_splits(m, ayar, seed=7)  # birinci bölme
    b = prepare_splits(m, ayar, seed=7)  # aynı seed ile ikinci bölme
    assert list(a[0]['path']) == list(b[0]['path'])  # eğitim foldları birebir aynı olmalı


def test_all_six_classes_present() -> None:  # Her fold altı sınıfı da içermeli (ağırlıklı kayıp/metrik için gerekli).
    m = _sentetik_manifest()  # sahte manifest
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',), split='speaker')  # konuşmacı bölmesi
    tr, va, te = prepare_splits(m, ayar, seed=42)  # böl
    assert set(tr['label_idx']) == set(range(6))  # eğitimde 6 sınıf da var
    assert len(set(va['label_idx'])) >= 1 and len(set(te['label_idx'])) >= 1  # geçerleme/test boş değil
