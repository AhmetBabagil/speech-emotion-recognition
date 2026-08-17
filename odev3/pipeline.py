'''Ödev 3 için uçtan uca doğrulama araması ve ayrılmış (held-out) test değerlendirmesi.

Bu dosya projenin "orkestra şefi"dir: diğer modüllerdeki parçaları doğru
sırayla ve doğru protokolle birbirine bağlar. Bir korpus için tam akış:

1. Konuşmacı-bağımsız 70/15/15 bölmelerini kur (Ödev 1-2 ile aynı, seed=42).
2. Eğitim + doğrulama özniteliklerini yükle, eğitim istatistikleriyle
   standardize et (test verisi bu aşamada BİLEREK yüklenmez).
3. Hiperparametre araması: tarama (screening) -> yerel iyileştirme
   (refinement) -> çok tohumlu kararlılık (stability). Her deneme erken
   durdurmayla eğitilir; en iyisi doğrulama macro-F1'e göre seçilir.
4. Doğrulama olasılıklarıyla sıcaklık kalibrasyonu öğren.
5. ANCAK ŞİMDİ test verisini yükle; seçilen modeli test üzerinde BİR KEZ
   değerlendir (metrikler + bootstrap belirsizliği + kalibrasyon raporu).
6. Tüm çıktıları (CSV, JSON, PNG, checkpoint) diske yaz.

Ek olarak, uzun aramalar kesintiye dayanıklıdır: her denemeden sonra arama
durumu (search_state.pt) atomik olarak kaydedilir ve uyumlu bir durumla
yeniden başlatılınca kaldığı yerden devam edilir.
'''

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from odev3.calibration import (
    classification_calibration_report,
    fit_temperature,
    temperature_scale_probabilities,
)
from odev3.dataset import FeatureStandardizer, load_feature_matrix
from odev3.features_melspec import DEFAULT_CONFIG, MelSpecConfig
from odev3.model import MLP, MLPConfig, count_parameters
from odev3.search_space import refinement_space, search_space
from odev3.training import (
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from odev3.uncertainty import bootstrap_metric_intervals
from ser.config import Config
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES
from ser.data import prepare_splits
from ser.utils import ensure_dir, get_device, get_logger, set_seed


# Proje genelinde ortak günlükçü (logger) ve deneyin ana tohumu.
# SEED=42 tüm ödevlerde aynıdır; böylece bölmeler ve sonuçlar karşılaştırılabilir.
log = get_logger('odev3.mlp')
SEED = 42
CORPUS_DISPLAY_NAMES = {'cremad': 'CREMA-D', 'meld': 'MELD'}


def _json_default(value: Any) -> Any:
    # json.dumps'ın tanımadığı tipleri (numpy skalerleri, diziler, Path)
    # düz Python karşılıklarına çeviren yardımcı. Bilinmeyen tiplerde
    # TypeError fırlatmak json protokolünün beklediği davranıştır.
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def _write_json(path: str | Path, payload: Any) -> None:
    # JSON yazımını tek yerde standartlaştırır: klasörü oluştur, 2 boşluk
    # girinti, Türkçe karakterleri kaçış dizisine çevirme (ensure_ascii=False),
    # UTF-8 kodlama. Tüm .json çıktıları bu fonksiyondan geçer.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding='utf-8',
    )


def _manifest_identity(path: str | Path) -> dict[str, Any]:
    # Manifest dosyasının kimliği: tam yol + boyut + değişiklik zamanı.
    # Arama imzasına (aşağıda) girer; manifest değişirse kayıtlı arama durumu
    # otomatik olarak geçersiz sayılır — bayat verilerle devam edilmez.
    manifest_path = Path(path)
    identity: dict[str, Any] = {'path': str(manifest_path.resolve())}
    if manifest_path.is_file():
        stat = manifest_path.stat()
        identity.update({'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return identity


def _search_signature(
    *,
    corpus: str,
    manifest_path: str | Path,
    grid_mode: str,
    max_epochs: int,
    feature_config: MelSpecConfig,
    limit_per_split: int | None,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    # Arama "imzası": sonucu etkileyebilecek TÜM ayarların anlık görüntüsü.
    # Devam etme (resume) mekanizması, diskteki search_state.pt yalnızca bu
    # imzayla birebir eşleşiyorsa kullanılır; tek bir ayar bile değişmişse
    # eski denemeler güvenilmez sayılıp arama baştan yapılır.
    return {
        'corpus': corpus,
        'manifest': _manifest_identity(manifest_path),
        'grid_mode': grid_mode,
        'max_epochs': max_epochs,
        'feature_config': asdict(feature_config),
        'limit_per_split': limit_per_split,
        'device': str(device),
        'amp': bool(amp and device.type == 'cuda'),
        'seed': SEED,
    }


def _config_json(config: MLPConfig) -> str:
    # Konfigürasyonun kanonik (deterministik) JSON hali: anahtarlar sıralı,
    # boşluk yok. Aynı ayarlar her zaman aynı metni üretir; kimlik ve anahtar
    # üretiminin temelidir.
    return json.dumps(config.to_dict(), sort_keys=True, separators=(',', ':'))


def _config_id(config: MLPConfig) -> str:
    # Kanonik JSON'un SHA-1 özetinin ilk 12 karakteri: CSV'lerde insan
    # tarafından okunabilir kısalıkta, pratikte çakışmayan bir konfig kimliği.
    return hashlib.sha1(_config_json(config).encode('utf-8')).hexdigest()[:12]


def _trial_key(stage: str, config: MLPConfig, trial_seed: int) -> str:
    # Bir denemenin benzersiz anahtarı: aşama + tohum + konfig. Aynı konfig
    # farklı aşamada ya da farklı tohumla AYRI bir deneme sayılır (örn.
    # stability aşamasında aynı konfig üç tohumla üç kez koşar).
    config_json = _config_json(config)
    return f'{stage}|{trial_seed}|{config_json}'


def _serialize_best_bundle(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    # "En iyi deneme paketi"ni diske yazılabilir hale getirir. Paket; skor
    # satırını, konfigürasyonu, eğitim geçmişini, doğrulama metriklerini ve
    # model ağırlıklarını içerir. MLPConfig nesnesi doğrudan pickle'lamak
    # yerine sözlüğe çevrilir — sürüm uyumluluğu için daha güvenli.
    if bundle is None:
        return None
    return {
        'row': bundle['row'],
        'config': bundle['config'].to_dict(),
        'history': bundle['history'],
        'validation_metrics': bundle['validation_metrics'],
        'state_dict': bundle['state_dict'],
    }


def _restore_best_bundle(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    # _serialize_best_bundle'ın tersi: diskten okunan sözlüğü bellekteki
    # biçime döndürür (config sözlüğü tekrar MLPConfig olur ve doğrulanır).
    if payload is None:
        return None
    return {
        'row': payload['row'],
        'config': MLPConfig.from_dict(payload['config']),
        'history': payload['history'],
        'validation_metrics': payload['validation_metrics'],
        'state_dict': payload['state_dict'],
    }


def _save_search_state(
    path: Path,
    *,
    signature: dict[str, Any],
    rows: list[dict[str, Any]],
    completed_keys: set[str],
    best_bundle: dict[str, Any] | None,
    refinement_base: MLPConfig | None,
    stability_base: MLPConfig | None,
) -> None:
    # Aramanın o anki tam durumunu diske yazar. schema_version, format
    # değişirse eski dosyaların otomatik reddedilmesini sağlar. Kayıt önce
    # geçici dosyaya yapılır, sonra os.replace ile atomik olarak taşınır:
    # yazma sırasında elektrik kesilse bile diskte asla yarım durum kalmaz.
    payload = {
        'schema_version': 2,
        'signature': signature,
        'rows': rows,
        'completed_keys': sorted(completed_keys),
        'best_bundle': _serialize_best_bundle(best_bundle),
        'refinement_base': refinement_base.to_dict() if refinement_base else None,
        'stability_base': stability_base.to_dict() if stability_base else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _normalized_search_signature(signature: dict[str, Any]) -> dict[str, Any]:
    '''Davranışsal olarak eşdeğer eski arama durumları için sonradan eklenen varsayılanları doldurur.

    Geriye dönük uyumluluk hilesi: frame_strategy alanı koda sonradan eklendi.
    Ondan önce kaydedilmiş durumlar bu alanı içermez ama davranışları bugünkü
    'crop_pad' varsayılanıyla aynıdır. İki imzayı karşılaştırmadan önce eksik
    alanı varsayılanla doldurursak, eski ama geçerli durumlar boşuna
    çöpe atılmaz. (json çevrimi ayrıca tip farklarını da normalize eder:
    tuple/list, numpy/int gibi.)
    '''

    normalized = json.loads(json.dumps(signature, default=_json_default))
    normalized.setdefault('feature_config', {}).setdefault(
        'frame_strategy',
        'crop_pad',
    )
    return normalized


def _load_search_state(
    path: Path,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    # Kaydedilmiş arama durumunu okumaya çalışır; herhangi bir sorun varsa
    # (dosya yok, okunamıyor, imza uyuşmuyor) None döndürür ve arama sıfırdan
    # başlar. Yani devam mekanizması "en kötü ihtimalle baştan" ilkesiyle
    # tamamen güvenli tasarlanmıştır — asla hatalı durumla devam edilmez.
    if not path.is_file():
        return None
    try:
        # weights_only=True: pickle üzerinden rastgele kod çalıştırılmasını
        # engelleyen güvenli yükleme (yeni PyTorch sürümleri). Eski sürümler
        # bu parametreyi tanımaz ve TypeError verir; o zaman klasik yükleme
        # kullanılır.
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location='cpu')
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
        log.warning('Ignoring unreadable search state %s: %s', path, error)
        return None
    # İmza karşılaştırması normalize edilmiş halde yapılır (bkz. yukarıdaki
    # fonksiyon): yalnızca gerçekten aynı deney tanımına ait durum kabul edilir.
    saved_signature = payload.get('signature', {})
    signatures_match = _normalized_search_signature(
        saved_signature
    ) == _normalized_search_signature(signature)
    if payload.get('schema_version') != 2 or not signatures_match:
        log.warning('Ignoring incompatible search state: %s', path)
        return None
    return {
        'rows': list(payload.get('rows', [])),
        'completed_keys': set(payload.get('completed_keys', [])),
        'best_bundle': _restore_best_bundle(payload.get('best_bundle')),
        'refinement_base': (
            MLPConfig.from_dict(payload['refinement_base'])
            if payload.get('refinement_base')
            else None
        ),
        'stability_base': (
            MLPConfig.from_dict(payload['stability_base'])
            if payload.get('stability_base')
            else None
        ),
    }


def _splits_for(corpus: str, manifest_path: str | Path):
    '''Ödev 1 ve 2 ile birebir aynı, seed-42 konuşmacı-bağımsız bölmeleri yeniden kullanır.

    İki kritik nokta:
    - **Konuşmacı-bağımsız bölme**: Aynı konuşmacının kayıtları asla hem
      eğitimde hem testte bulunmaz. Aksi halde model duyguyu değil KİŞİNİN
      SESİNİ ezberleyerek şişirilmiş skorlar elde ederdi.
    - **Ödevler arası tutarlılık**: Aynı manifest + aynı seed + aynı oranlar
      (70/15/15) sayesinde Ödev 2'nin klasik modelleriyle bu MLP'nin test
      skorları adil biçimde karşılaştırılabilir.
    '''

    manifest = pd.read_csv(manifest_path)
    cfg = Config()
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = 'speaker'
    cfg.data.val_fraction = 0.15
    cfg.data.test_fraction = 0.15
    return prepare_splits(manifest, cfg.data, seed=SEED)


def _stratified_limit(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    '''Tanı (diagnostic) amaçlı bir katmanı, tüm sınıfları koruyarak deterministik küçültür.

    Duman testlerinde (--limit-per-split) tüm veriyle çalışmak gereksiz yavaş
    olur; ama rastgele N satır almak bir sınıfı tamamen dışarıda bırakabilir
    ve eğitim çöker (sınıf ağırlıkları sıfır sayım kabul etmez). Bu yüzden
    "tabakalı" küçültme yapılır: önce her sınıftan eşit pay alınır, kalan
    kontenjan havuzdan rastgele doldurulur, sıra karıştırılır. Sabit tohum
    sayesinde aynı limit her zaman aynı alt kümeyi üretir.
    '''

    # Limit yoksa ya da katman zaten limitin altındaysa dokunma (kopya döndür).
    if limit is None or limit >= len(frame):
        return frame.copy()
    if limit < NUM_CLASSES:
        raise ValueError(f'Fold limit must be at least {NUM_CLASSES}, got {limit}.')

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    # Her sınıfın garanti payı: limit // sınıf_sayısı (en az 1).
    per_class = max(1, limit // NUM_CLASSES)
    for label in range(NUM_CLASSES):
        candidates = frame.index[frame['label_idx'] == label].to_numpy()
        if len(candidates) == 0:
            raise ValueError(f'Fold has no examples for label {label}.')
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())

    # Bölme artığı yüzünden limit tam dolmadıysa kalan yerleri, henüz
    # seçilmemiş satırlardan rastgele tamamla.
    remaining = limit - len(selected)
    if remaining > 0:
        pool = frame.index[~frame.index.isin(selected)].to_numpy()
        take = min(remaining, len(pool))
        selected.extend(rng.choice(pool, size=take, replace=False).tolist())
    # Sınıf sınıf seçtiğimiz için liste sınıf bloklarına ayrılmış durumda;
    # karıştırarak yapay sıralamayı bozuyoruz.
    rng.shuffle(selected)
    return frame.loc[selected].reset_index(drop=True)


def _fold_details(frame: pd.DataFrame) -> dict[str, Any]:
    # Tek bir katmanın (fold) özeti: kayıt sayısı, benzersiz konuşmacı sayısı
    # ve sınıf başına örnek sayıları. Rapordaki "veri dağılımı" tabloları
    # buradan beslenir.
    counts = frame['label_idx'].value_counts().to_dict()
    return {
        'records': int(len(frame)),
        'speakers': int(frame['speaker'].astype(str).nunique()),
        'class_counts': {
            emotion: int(counts.get(index, 0))
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        },
    }


def _split_summary(train, validation, test) -> dict[str, Any]:
    # Bölme özetini üretirken aynı zamanda EN ÖNEMLİ güvenlik kontrolünü
    # yapar: üç katmanın konuşmacı kümeleri kesişiyor mu? Küme kesişimi
    # (set &) boş olmalı; değilse "konuşmacı sızıntısı" vardır ve deney
    # geçersizdir — devam etmek yerine hemen hata fırlatılır.
    speaker_sets = {
        'train': set(train['speaker'].astype(str)),
        'validation': set(validation['speaker'].astype(str)),
        'test': set(test['speaker'].astype(str)),
    }
    overlaps = {
        'train_validation': sorted(speaker_sets['train'] & speaker_sets['validation']),
        'train_test': sorted(speaker_sets['train'] & speaker_sets['test']),
        'validation_test': sorted(speaker_sets['validation'] & speaker_sets['test']),
    }
    if any(overlaps.values()):
        raise ValueError(f'Speaker leakage detected: {overlaps}.')
    # Boş kesişim listeleri de bilinçli olarak çıktıya yazılır: "sızıntı yok"
    # iddiasının kanıtı raporda görünür olur.
    return {
        'protocol': 'speaker-independent 70/15/15, seed=42',
        'train': _fold_details(train),
        'validation': _fold_details(validation),
        'test': _fold_details(test),
        'speaker_overlap': overlaps,
    }


def _plot_confusion(matrix: list[list[int]], path: str | Path, title: str) -> None:
    # Karışıklık (confusion) matrisini ısı haritası olarak çizer.
    # matplotlib importları fonksiyon içinde: modül yüklenirken değil, ancak
    # gerçekten grafik çizilecekse yüklensin (başlangıç süresi + bağımlılık).
    import matplotlib

    # 'Agg' arka ucu ekransız ortamda PNG üretebilmek için; pyplot'tan önce
    # seçilmek zorundadır.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Ham sayımlar yerine SATIR-normalize oranlar gösterilir: her hücre
    # "gerçek sınıfı X olanların yüzde kaçı Y tahmin edildi" anlamına gelir.
    # Böylece sınıf boyutları farklı olsa da hücreler karşılaştırılabilir.
    # np.divide'ın where parametresi boş satırlarda 0'a bölmeyi önler.
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    displayed = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )
    fig, axis = plt.subplots(figsize=(7.0, 5.8))
    # vmin/vmax sabit [0,1]: renk skalası tüm grafiklerde aynı anlama gelir.
    image = axis.imshow(displayed, cmap='Blues', vmin=0.0, vmax=1.0)
    axis.set_xticks(range(NUM_CLASSES), CANONICAL_EMOTIONS, rotation=45, ha='right')
    axis.set_yticks(range(NUM_CLASSES), CANONICAL_EMOTIONS)
    axis.set_xlabel('Tahmin edilen sınıf')
    axis.set_ylabel('Gerçek sınıf')
    axis.set_title(title)
    # Her hücrenin üzerine oranı sayı olarak yaz; koyu hücrelerde beyaz,
    # açık hücrelerde siyah yazı kullanarak okunabilirliği koru.
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = displayed[row, column]
            axis.text(
                column,
                row,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=8,
                color='white' if value > 0.5 else 'black',
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(history: list[dict[str, Any]], path: str | Path, title: str) -> None:
    # Öğrenme eğrileri: solda kayıp, sağda macro-F1 (eğitim vs doğrulama).
    # İki eğrinin makasının açılması (eğitim iyileşirken doğrulamanın
    # kötüleşmesi) aşırı öğrenmenin klasik görüntüsüdür; erken durdurmanın
    # neden gerektiği bu grafikte görülür.
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(epochs, [row['train_loss'] for row in history], label='Eğitim')
    axes[0].plot(epochs, [row['val_loss'] for row in history], label='Geçerleme')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ağırlıklı çapraz entropi')
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row['train_macro_f1'] for row in history],
        label='Eğitim',
    )
    axes[1].plot(
        epochs,
        [row['val_macro_f1'] for row in history],
        label='Geçerleme',
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro-F1')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reliability(
    before: dict[str, Any],
    after: dict[str, Any],
    path: str | Path,
    title: str,
) -> None:
    # Güvenilirlik diyagramı: solda "ortalama güven vs gerçek doğruluk"
    # eğrisi (45 derecelik kesikli çizgi = mükemmel kalibrasyon), sağda güven
    # dağılımı histogramı. Ham ve sıcaklık-ölçekli sonuçlar yan yana çizilir;
    # kalibrasyonun etkisi tek bakışta görülür.
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def nonempty_series(report: dict[str, Any]) -> tuple[list[float], list[float]]:
        # Boş dilimlerin (count=0) güven/doğruluk değeri None'dır; eğriye
        # dahil edilemezler, süzülüp atlanır.
        entries = [entry for entry in report['bins'] if entry['count'] > 0]
        return (
            [float(entry['mean_confidence']) for entry in entries],
            [float(entry['accuracy']) for entry in entries],
        )

    raw_confidence, raw_accuracy = nonempty_series(before)
    scaled_confidence, scaled_accuracy = nonempty_series(after)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axes[0].plot([0.0, 1.0], [0.0, 1.0], '--', color='0.45', label='İdeal')
    axes[0].plot(
        raw_confidence,
        raw_accuracy,
        'o-',
        label=f'Ham (ECE={before["expected_calibration_error"]:.3f})',
    )
    axes[0].plot(
        scaled_confidence,
        scaled_accuracy,
        's-',
        label=f'Ölçekli (ECE={after["expected_calibration_error"]:.3f})',
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel('Ortalama güven')
    axes[0].set_ylabel('Gerçek doğruluk')
    axes[0].set_title('Güvenilirlik eğrisi')
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    indices = np.arange(int(before['num_bins']))
    width = 0.38
    raw_counts = [int(entry['count']) for entry in before['bins']]
    scaled_counts = [int(entry['count']) for entry in after['bins']]
    labels = [
        f'{entry["lower"]:.1f}–{entry["upper"]:.1f}'
        for entry in before['bins']
    ]
    axes[1].bar(indices - width / 2, raw_counts, width, label='Ham')
    axes[1].bar(indices + width / 2, scaled_counts, width, label='Ölçekli')
    axes[1].set_xticks(indices, labels, rotation=45, ha='right')
    axes[1].set_xlabel('Güven aralığı')
    axes[1].set_ylabel('Örnek sayısı')
    axes[1].set_title('Güven dağılımı')
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _validation_row(
    trial_id: int,
    search_stage: str,
    trial_seed: int,
    config: MLPConfig,
    outcome,
    elapsed_seconds: float,
) -> dict[str, Any]:
    # Tek bir denemenin validation_results.csv'ye yazılacak "düz" satırı.
    # Konfigürasyonun her alanı ayrı sütuna açılır ki CSV'de filtrelemek ve
    # hiperparametre etkisi analizinde gruplamak kolay olsun. hidden_dims
    # tuple'ı '512-256' biçiminde metne çevrilir (CSV hücresinde liste olmaz).
    metrics = outcome.validation_metrics
    return {
        'trial': trial_id,
        'search_stage': search_stage,
        'seed': trial_seed,
        'config_id': _config_id(config),
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'patience': config.patience,
        'hidden_dims': '-'.join(str(value) for value in config.hidden_dims),
        'hidden_layers': len(config.hidden_dims),
        'activation': config.activation,
        'batch_norm': config.batch_norm,
        'dropout': config.dropout,
        'weight_decay': config.weight_decay,
        'parameters': count_parameters(outcome.model),
        'best_epoch': outcome.best_epoch,
        'epochs_trained': outcome.epochs_trained,
        'stopped_early': outcome.stopped_early,
        'val_loss': outcome.validation_loss,
        'val_accuracy': metrics['accuracy'],
        'val_balanced_accuracy': metrics['balanced_accuracy'],
        'val_macro_f1': metrics['macro_f1'],
        'val_weighted_f1': metrics['weighted_f1'],
        'elapsed_seconds': elapsed_seconds,
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    # "En iyi deneme" seçiminin sıralama anahtarı. Python tuple'ları eleman
    # eleman karşılaştırır: önce macro-F1, eşitse dengeli doğruluk, o da
    # eşitse DÜŞÜK doğrulama kaybı (eksi işaretiyle "büyük olan kazanır"
    # kuralına çevrilir). Tamamen deterministik bir seçim kuralıdır.
    return (
        float(row['val_macro_f1']),
        float(row['val_balanced_accuracy']),
        -float(row['val_loss']),
    )


def _feature_method(config: MelSpecConfig) -> str:
    # Öznitelik boru hattının insan tarafından okunabilir tek satırlık özeti;
    # result.json'a ve rapora "yöntem" alanı olarak yazılır.
    return (
        'librosa.feature.melspectrogram -> log dB -> '
        f'{config.n_mels}x{config.n_frames} -> '
        f'{config.frame_strategy} -> flatten'
    )


def run_corpus(
    corpus: str,
    *,
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    grid_mode: str,
    max_epochs: int,
    device: torch.device,
    feature_config: MelSpecConfig = DEFAULT_CONFIG,
    feature_workers: int = 1,
    loader_workers: int = 0,
    amp: bool = True,
    limit_per_split: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    '''Doğrulama verisinde arama yapar, seçilen modeli test üzerinde BİR KEZ değerlendirir.

    Bu fonksiyon tek bir korpusun tüm deneyini uçtan uca yürütür. Altın
    kural: test kümesi model seçiminin hiçbir aşamasında kullanılmaz; ancak
    kazanan kesinleştikten sonra, tek bir değerlendirme için yüklenir. Bu
    disiplin sayesinde raporlanan test skoru gerçek bir "genelleme" ölçüsüdür,
    seçim sürecinin yan ürünü değildir.
    '''

    # ------------------------------------------------------------------
    # 1) Kurulum: tohum, klasörler ve veri bölmeleri.
    # ------------------------------------------------------------------
    set_seed(SEED)
    corpus_dir = ensure_dir(Path(output_root) / corpus)
    history_dir = ensure_dir(corpus_dir / 'histories')
    cache_dir = Path(cache_root) / corpus
    train_frame, validation_frame, test_frame = _splits_for(corpus, manifest_path)
    # Tanı modunda katmanları küçült; her katmana FARKLI tohum verilir ki
    # alt kümeler birbirinden bağımsız örneklensin.
    train_frame = _stratified_limit(train_frame, limit_per_split, SEED)
    validation_frame = _stratified_limit(validation_frame, limit_per_split, SEED + 1)
    test_frame = _stratified_limit(test_frame, limit_per_split, SEED + 2)
    # Bölme özeti hem konuşmacı sızıntısını denetler hem de kanıt olarak
    # diske yazılır.
    split_summary = _split_summary(train_frame, validation_frame, test_frame)
    _write_json(corpus_dir / 'split_summary.json', split_summary)
    log.info(
        '[%s] train=%d validation=%d test=%d | mode=%s',
        corpus,
        len(train_frame),
        len(validation_frame),
        len(test_frame),
        grid_mode,
    )

    # ------------------------------------------------------------------
    # 2) Öznitelikler: yalnızca eğitim + doğrulama.
    # Test öznitelikleri ve etiketleri, doğrulama araması tek bir aday
    # seçene kadar BİLEREK yüklenmez — protokolün temel taşı budur.
    # ------------------------------------------------------------------
    train_features, train_labels = load_feature_matrix(
        train_frame,
        cache_dir,
        feature_config,
        workers=feature_workers,
        description=f'{corpus} eğitim',
    )
    validation_features, validation_labels = load_feature_matrix(
        validation_frame,
        cache_dir,
        feature_config,
        workers=feature_workers,
        description=f'{corpus} geçerleme',
    )
    # z-score parametreleri SADECE eğitim verisinden öğrenilir; doğrulama
    # yalnızca dönüştürülür (veri sızıntısını önleme). Sınıf ağırlıkları da
    # eğitim dağılımından hesaplanır.
    standardizer = FeatureStandardizer.fit(train_features)
    train_features = standardizer.transform(train_features)
    validation_features = standardizer.transform(validation_features)
    class_weights = inverse_frequency_weights(train_labels, NUM_CLASSES)

    # ------------------------------------------------------------------
    # 3) Arama durumu: imzayı kur, varsa uyumlu kayıtlı durumu geri yükle.
    # ------------------------------------------------------------------
    screening_candidates = search_space(grid_mode)
    state_path = corpus_dir / 'search_state.pt'
    signature = _search_signature(
        corpus=corpus,
        manifest_path=manifest_path,
        grid_mode=grid_mode,
        max_epochs=max_epochs,
        feature_config=feature_config,
        limit_per_split=limit_per_split,
        device=device,
        amp=amp,
    )
    saved_state = _load_search_state(state_path, signature) if resume else None
    if saved_state:
        # Uyumlu kayıt bulundu: tamamlanmış denemeler, en iyi paket ve aşama
        # tabanları (refinement/stability başlangıç konfigleri) geri gelir.
        rows: list[dict[str, Any]] = saved_state['rows']
        completed_keys: set[str] = saved_state['completed_keys']
        best_bundle: dict[str, Any] | None = saved_state['best_bundle']
        refinement_base: MLPConfig | None = saved_state['refinement_base']
        stability_base: MLPConfig | None = saved_state['stability_base']
        log.info('[%s] resuming %d completed validation trials', corpus, len(rows))
    else:
        # Temiz başlangıç: hiç deneme yok, en iyi henüz belirsiz.
        rows = []
        completed_keys = set()
        best_bundle = None
        refinement_base = None
        stability_base = None
    stage_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 4) Arama motoru: run_stage, bir aday listesini sırayla eğitir.
    # İç fonksiyon (closure) olarak tanımlanır çünkü öznitelik matrisleri,
    # rows, completed_keys gibi ortak duruma doğrudan erişmesi gerekir;
    # bunları parametre olarak taşımak imzayı gereksiz şişirirdi.
    # ------------------------------------------------------------------
    def run_stage(
        stage: str,
        candidates: list[MLPConfig],
        trial_seed: int = SEED,
    ) -> None:
        # nonlocal: best_bundle dış kapsamda yaşar; iç fonksiyon onu
        # yeniden atayabilsin diye bildirilir (okumak için gerekmezdi).
        nonlocal best_bundle
        stage_counts[stage] = stage_counts.get(stage, 0) + len(candidates)
        for stage_index, candidate in enumerate(candidates, start=1):
            # Deneme anahtarı (aşama|tohum|konfig) daha önce tamamlanmışsa
            # eğitimi atla — devam etme mekanizmasının çekirdeği.
            key = _trial_key(stage, candidate, trial_seed)
            if key in completed_keys:
                log.info(
                    '[%s] skipping completed %s trial %d/%d',
                    corpus,
                    stage,
                    stage_index,
                    len(candidates),
                )
                continue
            # Global deneme numarası: aşamalardan bağımsız, 1'den itibaren
            # artan tek bir sayaç (CSV'de benzersiz kimlik görevi görür).
            trial_id = len(rows) + 1
            log.info(
                '[%s] %s trial %d/%d | global trial %d | %s',
                corpus,
                stage,
                stage_index,
                len(candidates),
                trial_id,
                candidate.to_dict(),
            )
            started = time.perf_counter()
            outcome = train_with_early_stopping(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                candidate,
                input_dim=feature_config.vector_size,
                num_classes=NUM_CLASSES,
                device=device,
                max_epochs=max_epochs,
                seed=trial_seed,
                num_workers=loader_workers,
                amp=amp,
            )
            elapsed = time.perf_counter() - started
            row = _validation_row(
                trial_id,
                stage,
                trial_seed,
                candidate,
                outcome,
                elapsed,
            )
            rows.append(row)
            # Her denemenin epoch-epoch geçmişi ayrı CSV'ye; tüm satırların
            # o anki hali de partial CSV'ye yazılır (izleme + güvenlik ağı).
            pd.DataFrame(outcome.history).to_csv(
                history_dir / f'trial_{trial_id:03d}.csv', index=False
            )
            pd.DataFrame(rows).to_csv(
                corpus_dir / 'validation_results.partial.csv', index=False
            )
            log.info(
                '[%s] trial %d | best epoch=%d, val macro-F1=%.4f',
                corpus,
                trial_id,
                outcome.best_epoch,
                row['val_macro_f1'],
            )

            # Yeni deneme, mevcut en iyiden daha iyiyse (bkz. _selection_key)
            # "en iyi paket" güncellenir: skor satırı, konfig, geçmiş ve
            # model ağırlıklarının CPU kopyası saklanır. Ağırlıkları hemen
            # saklamak gerekir; sıradaki deneme aynı GPU belleğini kullanacak.
            if best_bundle is None or _selection_key(row) > _selection_key(
                best_bundle['row']
            ):
                best_bundle = {
                    'row': row.copy(),
                    'config': candidate,
                    'history': list(outcome.history),
                    'validation_metrics': outcome.validation_metrics,
                    'state_dict': {
                        name: value.detach().cpu().clone()
                        for name, value in outcome.model.state_dict().items()
                    },
                }
            # Denemeyi tamamlandı olarak işaretle ve arama durumunu HEMEN
            # diske yaz: bu satırdan sonra kesinti olsa bile deneme kayıpsız.
            completed_keys.add(key)
            _save_search_state(
                state_path,
                signature=signature,
                rows=rows,
                completed_keys=completed_keys,
                best_bundle=best_bundle,
                refinement_base=refinement_base,
                stability_base=stability_base,
            )
            # Model/geçmiş referanslarını bırak ve CUDA önbelleğini boşalt:
            # onlarca deneme art arda koşarken GPU belleği birikmesin.
            del outcome
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5) Arama planı. 'report' modunda üç aşama koşar:
    #    screening (ızgara) -> refinement (kazananın komşuları) ->
    #    stability (kazananı 2 ek tohumla tekrar eğit).
    #    Diğer modlarda tek aşama vardır ve adıyla anılır (quick/full).
    # ------------------------------------------------------------------
    initial_stage = 'screening' if grid_mode == 'report' else grid_mode
    run_stage(initial_stage, screening_candidates)

    refinement_candidates: list[MLPConfig] = []
    if grid_mode == 'report':
        if best_bundle is None:
            raise RuntimeError(f'Screening produced no valid trial for {corpus}.')
        # refinement_base bir kez sabitlenir ve duruma kaydedilir. Neden?
        # Devam eden bir çalıştırmada screening'in kazananı değişmiş olsa
        # bile refinement HEP aynı taban etrafında üretilmeli; yoksa yarıda
        # kalan aşama farklı aday listesiyle devam eder ve anahtarlar şaşardı.
        if refinement_base is None:
            refinement_base = best_bundle['config']
            _save_search_state(
                state_path,
                signature=signature,
                rows=rows,
                completed_keys=completed_keys,
                best_bundle=best_bundle,
                refinement_base=refinement_base,
                stability_base=stability_base,
            )
        # Screening'de zaten denenen adaylar refinement listesinden çıkarılır
        # (exclude) — aynı konfigi iki kez eğitmek zaman kaybı olurdu.
        refinement_candidates = refinement_space(
            refinement_base,
            exclude=screening_candidates,
        )
        run_stage('refinement', refinement_candidates)
        if best_bundle is None:
            raise RuntimeError(f'Refinement produced no valid trial for {corpus}.')
        # stability_base da aynı nedenle bir kez sabitlenir: iki aşamanın
        # genel kazananı, farklı tohumlarla tekrar eğitilecek konfigdir.
        if stability_base is None:
            stability_base = best_bundle['config']
            _save_search_state(
                state_path,
                signature=signature,
                rows=rows,
                completed_keys=completed_keys,
                best_bundle=best_bundle,
                refinement_base=refinement_base,
                stability_base=stability_base,
            )
        # Kararlılık aşaması: aynı konfig, iki FARKLI tohum (ana tohum SEED
        # ile zaten eğitildi). Skorların tohumdan tohuma ne kadar oynadığı,
        # sonucun şansa ne kadar bağlı olduğunu gösterir.
        run_stage('stability', [stability_base], trial_seed=SEED + 101)
        run_stage('stability', [stability_base], trial_seed=SEED + 202)

    if best_bundle is None:
        raise RuntimeError(f'No successful validation trial for {corpus}.')

    # ------------------------------------------------------------------
    # 6) Arama bitti: sonuç tablosunu sırala, kazananı işaretle ve yaz.
    # ------------------------------------------------------------------
    ranked = pd.DataFrame(rows).sort_values(
        ['val_macro_f1', 'val_balanced_accuracy', 'val_loss'],
        ascending=[False, False, True],
    )
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    ranked['selected'] = ranked['trial'] == int(best_bundle['row']['trial'])
    ranked.to_csv(corpus_dir / 'validation_results.csv', index=False)

    # Kazanan modeli, saklanan ağırlıklardan yeniden kur. (Aramadaki model
    # nesneleri bellekten atıldı; elimizde yalnızca state_dict kopyası var.)
    best_config: MLPConfig = best_bundle['config']
    model = MLP(feature_config.vector_size, NUM_CLASSES, best_config)
    model.load_state_dict(best_bundle['state_dict'])
    model.to(device)

    # ------------------------------------------------------------------
    # 7) Kalibrasyon: sıcaklık, DOĞRULAMA olasılıkları üzerinde öğrenilir.
    # Testten önce öğrenilmesi şart — kalibrasyon da bir "model seçimi"dir
    # ve test verisine bakamaz.
    # ------------------------------------------------------------------
    _, _, validation_probabilities = evaluate_arrays(
        model,
        validation_features,
        validation_labels,
        class_weights=class_weights,
        device=device,
        batch_size=max(best_config.batch_size, 256),
        num_workers=loader_workers,
    )
    temperature_fit = fit_temperature(
        validation_probabilities,
        validation_labels,
    )

    # ------------------------------------------------------------------
    # 8) TEST: ayrılmış test kayıtlarının yüklendiği/kullanıldığı İLK nokta
    # burasıdır. Model ve kalibrasyon çoktan kesinleşti; test artık yalnızca
    # tarafsız bir ölçüm.
    # ------------------------------------------------------------------
    test_features, test_labels = load_feature_matrix(
        test_frame,
        cache_dir,
        feature_config,
        workers=feature_workers,
        description=f'{corpus} test',
    )
    test_features = standardizer.transform(test_features)
    test_loss, test_metrics, probabilities = evaluate_arrays(
        model,
        test_features,
        test_labels,
        class_weights=class_weights,
        device=device,
        batch_size=max(best_config.batch_size, 256),
        num_workers=loader_workers,
    )
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    # Test metriklerinin belirsizliği: tabakalı bootstrap ile %95 güven
    # aralıkları (bkz. odev3/uncertainty.py).
    test_uncertainty = bootstrap_metric_intervals(
        test_labels,
        predictions,
        iterations=2000,
        confidence=0.95,
        seed=SEED,
    )
    # Ham (kalibre edilmemiş) test olasılıklarının kalibrasyon raporu.
    test_calibration = classification_calibration_report(
        probabilities,
        test_labels,
        bins=10,
    )
    # Doğrulamada öğrenilen sıcaklığı hem doğrulama hem test olasılıklarına
    # uygula; öncesi/sonrası raporları yan yana kaydedilecek.
    temperature = float(temperature_fit['temperature'])
    calibrated_validation_probabilities = temperature_scale_probabilities(
        validation_probabilities,
        temperature,
    )
    calibrated_probabilities = temperature_scale_probabilities(
        probabilities,
        temperature,
    )
    # Güvenlik doğrulaması: sıcaklık ölçekleme tanımı gereği argmax'ı
    # değiştiremez. Değiştirdiyse kodda ciddi bir hata var demektir; sessizce
    # devam etmek yerine çök.
    calibrated_predictions = calibrated_probabilities.argmax(axis=1)
    if not np.array_equal(calibrated_predictions, predictions):
        raise RuntimeError('Temperature scaling changed predicted class indices.')
    # Kalibrasyonun tam hikayesi: öğrenme detayları + doğrulama ve test için
    # öncesi/sonrası raporlar. Rapor bu sözlükten tablo üretir.
    temperature_scaling = {
        'fit': temperature_fit,
        'validation': {
            'before': classification_calibration_report(
                validation_probabilities,
                validation_labels,
                bins=10,
            ),
            'after': classification_calibration_report(
                calibrated_validation_probabilities,
                validation_labels,
                bins=10,
            ),
        },
        'test': {
            'before': test_calibration,
            'after': classification_calibration_report(
                calibrated_probabilities,
                test_labels,
                bins=10,
            ),
        },
        'class_predictions_preserved': True,
    }

    # ------------------------------------------------------------------
    # 9) Örnek bazında tahmin dökümü: her test kaydı için gerçek/tahmin
    # etiketi, güven ve tüm sınıf olasılıkları (ham + kalibre). Hata analizi
    # ("model en çok neyi neyle karıştırıyor?") bu CSV üzerinden yapılır.
    # ------------------------------------------------------------------
    prediction_frame = test_frame[
        ['path', 'speaker', 'emotion', 'label_idx']
    ].reset_index(drop=True).copy()
    prediction_frame['predicted_idx'] = predictions
    prediction_frame['predicted_emotion'] = [
        CANONICAL_EMOTIONS[int(value)] for value in predictions
    ]
    prediction_frame['confidence'] = confidences
    prediction_frame['calibrated_confidence'] = calibrated_probabilities.max(axis=1)
    for index, emotion in enumerate(CANONICAL_EMOTIONS):
        prediction_frame[f'prob_{emotion}'] = probabilities[:, index]
        prediction_frame[f'calibrated_prob_{emotion}'] = calibrated_probabilities[
            :, index
        ]
    prediction_frame.to_csv(corpus_dir / 'test_predictions.csv', index=False)
    uncertainty_path = corpus_dir / 'test_uncertainty.json'
    _write_json(uncertainty_path, test_uncertainty)
    calibration_path = corpus_dir / 'test_calibration.json'
    _write_json(calibration_path, test_calibration)
    temperature_scaling_path = corpus_dir / 'temperature_scaling.json'
    _write_json(temperature_scaling_path, temperature_scaling)

    # ------------------------------------------------------------------
    # 10) Teslim edilebilir checkpoint: modeli daha sonra TEK dosyadan
    # eksiksiz kurabilmek için gereken HER ŞEY buraya yazılır — ağırlıklar,
    # mimari konfigi, öznitelik ayarları, standardizasyon parametreleri,
    # sınıf ağırlıkları ve kalibrasyon sıcaklığı. (load_saved_model bu
    # dosyayı okur.)
    # ------------------------------------------------------------------
    checkpoint = {
        'schema_version': 2,
        'assignment': 3,
        'corpus': corpus,
        'model_type': 'PyTorch MLP from scratch',
        'input_dim': feature_config.vector_size,
        'num_classes': NUM_CLASSES,
        'class_names': list(CANONICAL_EMOTIONS),
        'feature_config': asdict(feature_config),
        'model_config': best_config.to_dict(),
        'model_state_dict': best_bundle['state_dict'],
        'standardizer_mean': torch.from_numpy(standardizer.mean.copy()),
        'standardizer_scale': torch.from_numpy(standardizer.scale.copy()),
        'class_weights': class_weights.clone(),
        'seed': int(best_bundle['row']['seed']),
        'selection_metric': 'validation macro-F1',
        'best_epoch': int(best_bundle['row']['best_epoch']),
        'probability_calibration': {
            'method': 'temperature scaling',
            'temperature': temperature,
            'fitted_on': 'validation probabilities',
        },
    }
    torch.save(checkpoint, corpus_dir / 'best_model.pt')

    # ------------------------------------------------------------------
    # 11) Görseller: en iyi modelin öğrenme eğrileri, test karışıklık
    # matrisi ve kalibrasyon güvenilirlik diyagramı.
    # ------------------------------------------------------------------
    history_path = corpus_dir / 'best_training_history.csv'
    pd.DataFrame(best_bundle['history']).to_csv(history_path, index=False)
    _plot_history(
        best_bundle['history'],
        corpus_dir / 'best_training_history.png',
        f'{CORPUS_DISPLAY_NAMES.get(corpus, corpus.upper())} - en iyi MLP eğitim süreci',
    )
    _plot_confusion(
        test_metrics['confusion_matrix'],
        corpus_dir / 'test_confusion_matrix.png',
        f'{CORPUS_DISPLAY_NAMES.get(corpus, corpus.upper())} - MLP test karmaşıklık matrisi',
    )
    reliability_path = corpus_dir / 'test_reliability_diagram.png'
    _plot_reliability(
        temperature_scaling['test']['before'],
        temperature_scaling['test']['after'],
        reliability_path,
        f'{CORPUS_DISPLAY_NAMES.get(corpus, corpus.upper())} - test olasılık kalibrasyonu',
    )

    # ------------------------------------------------------------------
    # 12) Kararlılık özeti: kazanan konfigürasyonun tüm tohumlardaki
    # skorlarını (ana arama tohumu + iki stability tohumu) ortalama/std/
    # min/max olarak toparlar. config_id ile eşleştirme yapılır çünkü aynı
    # konfig birden çok aşamada satır üretmiş olabilir.
    # ------------------------------------------------------------------
    stability_summary = None
    if grid_mode == 'report' and stability_base is not None:
        stability_id = _config_id(stability_base)
        stability_rows = [row for row in rows if row['config_id'] == stability_id]
        stability_summary = {
            'config_id': stability_id,
            'config': stability_base.to_dict(),
            'seeds': [int(row['seed']) for row in stability_rows],
            'runs': len(stability_rows),
            'val_macro_f1_mean': float(
                np.mean([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_macro_f1_std': float(
                np.std([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_macro_f1_min': float(
                np.min([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_macro_f1_max': float(
                np.max([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_balanced_accuracy_mean': float(
                np.mean([row['val_balanced_accuracy'] for row in stability_rows])
            ),
            'val_accuracy_mean': float(
                np.mean([row['val_accuracy'] for row in stability_rows])
            ),
        }

    # ------------------------------------------------------------------
    # 13) Nihai sonuç sözlüğü: deneyin TÜM hikayesi tek yerde — protokol,
    # veri, kazanan model, doğrulama/test metrikleri, belirsizlik,
    # kalibrasyon ve üretilen dosyaların yolları. result.json'a yazılır ve
    # rapor üretimi ile testler bu dosyayı okur.
    # ------------------------------------------------------------------
    result = {
        'corpus': corpus,
        'manifest': str(manifest_path),
        'grid_mode': grid_mode,
        'diagnostic_limit_per_split': limit_per_split,
        'seed': SEED,
        'selection_metric': 'validation macro-F1',
        'search_protocol': (
            'screening -> local refinement -> multi-seed stability'
            if grid_mode == 'report'
            else grid_mode
        ),
        'stability_seeds': [SEED, SEED + 101, SEED + 202]
        if grid_mode == 'report'
        else [SEED],
        'num_trials': len(rows),
        'search_stages': stage_counts,
        'stability': stability_summary,
        'feature': {
            **asdict(feature_config),
            'vector_size': feature_config.vector_size,
            'method': _feature_method(feature_config),
            'pca': False,
        },
        'preprocessing': {
            'normalization': 'per-dimension z-score',
            'normalizer_fit': 'training fold only',
        },
        'split': split_summary,
        'training_class_weights': {
            emotion: float(class_weights[index])
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        },
        'best_trial': int(best_bundle['row']['trial']),
        'selected_seed': int(best_bundle['row']['seed']),
        'best_config': best_config.to_dict(),
        'model_parameters': count_parameters(model),
        'best_epoch': int(best_bundle['row']['best_epoch']),
        'epochs_trained': int(best_bundle['row']['epochs_trained']),
        'stopped_early': bool(best_bundle['row']['stopped_early']),
        'validation_loss': float(best_bundle['row']['val_loss']),
        'validation': best_bundle['validation_metrics'],
        'test_loss': float(test_loss),
        'test': test_metrics,
        'test_uncertainty': test_uncertainty,
        'test_calibration': test_calibration,
        'temperature_scaling': temperature_scaling,
        'artifacts': {
            'model': str(corpus_dir / 'best_model.pt'),
            'validation_results': str(corpus_dir / 'validation_results.csv'),
            'history': str(history_path),
            'confusion_matrix': str(corpus_dir / 'test_confusion_matrix.png'),
            'predictions': str(corpus_dir / 'test_predictions.csv'),
            'uncertainty': str(uncertainty_path),
            'calibration': str(calibration_path),
            'temperature_scaling': str(temperature_scaling_path),
            'reliability_diagram': str(reliability_path),
            'search_state': str(state_path),
        },
    }
    _write_json(corpus_dir / 'result.json', result)
    log.info(
        '[%s] TEST | accuracy=%.4f balanced=%.4f macro-F1=%.4f',
        corpus,
        test_metrics['accuracy'],
        test_metrics['balanced_accuracy'],
        test_metrics['macro_f1'],
    )
    log.info(
        '[%s] CALIBRATION | T=%.4f val-NLL %.4f->%.4f test-ECE %.4f->%.4f',
        corpus,
        temperature,
        temperature_fit['validation_nll_before'],
        temperature_fit['validation_nll_after'],
        test_calibration['expected_calibration_error'],
        temperature_scaling['test']['after']['expected_calibration_error'],
    )
    return result


def _write_model_comparison(
    results: dict[str, dict[str, Any]],
    prior_results_path: str | Path,
    output_root: str | Path,
) -> pd.DataFrame:
    # Ödevler arası karşılaştırma tablosu: Ödev 2'nin test sonuçları CSV'si
    # okunur (varsa), MLP'nin satırları eklenir ve korpus içi macro-F1'e göre
    # sıralanmış tek bir model_comparison.csv üretilir. Böylece raporda
    # "MLP klasik modellere göre nerede duruyor?" sorusu tek tablodan okunur.
    output_root = ensure_dir(output_root)
    columns = [
        'corpus',
        'model',
        'feature_dim',
        'pca_dim',
        'params',
        'test_accuracy',
        'test_balanced_accuracy',
        'test_macro_f1',
        'test_weighted_f1',
    ]
    # Önceki ödevin dosyası yoksa (örn. duman testi ortamı) boş bir çerçeveyle
    # başlanır; tablo yalnızca MLP satırlarını içerir.
    prior_path = Path(prior_results_path)
    if prior_path.is_file():
        comparison = pd.read_csv(prior_path)
    else:
        comparison = pd.DataFrame(columns=columns)

    new_rows = []
    for corpus, result in results.items():
        test = result['test']
        new_rows.append(
            {
                'corpus': corpus,
                'model': 'MLP (Ödev 3)',
                'feature_dim': result['feature']['vector_size'],
                'pca_dim': 'none',
                'params': json.dumps(
                    result['best_config'], ensure_ascii=False, sort_keys=True
                ),
                'test_accuracy': test['accuracy'],
                'test_balanced_accuracy': test['balanced_accuracy'],
                'test_macro_f1': test['macro_f1'],
                'test_weighted_f1': test['weighted_f1'],
            }
        )
    comparison = pd.concat([comparison[columns], pd.DataFrame(new_rows)], ignore_index=True)
    comparison = comparison.sort_values(
        ['corpus', 'test_macro_f1'], ascending=[True, False]
    ).reset_index(drop=True)
    comparison.to_csv(output_root / 'model_comparison.csv', index=False)
    return comparison


def run_all(
    *,
    manifest_path: str | Path = 'odev1/manifest_subset.csv',
    cache_root: str | Path = 'data/cache/odev3_melspec',
    output_root: str | Path = 'odev3/outputs',
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    grid_mode: str = 'report',
    max_epochs: int = 60,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
    amp: bool = True,
    limit_per_split: int | None = None,
    prior_results_path: str | Path = 'odev2/outputs/test_comparison_with_knn.csv',
    resume: bool = True,
    feature_config: MelSpecConfig = DEFAULT_CONFIG,
    feature_configs: dict[str, MelSpecConfig] | None = None,
) -> dict[str, dict[str, Any]]:
    '''İstenen her veri kümesi için BAĞIMSIZ aramalar çalıştırır.

    "Bağımsız" vurgusu önemli: CREMA-D ve MELD ayrı deneylerdir — ayrı
    bölmeler, ayrı arama, ayrı kazanan model. Bir kümenin sonucu diğerini
    hiçbir şekilde etkilemez. Bu fonksiyon ayrıca korpus başına farklı
    öznitelik ayarına izin verir (feature_configs sözlüğü genel
    feature_config'i ezer) ve sonunda karşılaştırma tablosu + genel
    summary.json üretir.
    '''

    # --- Girdi doğrulamaları ve ortak kurulum ---
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}.')
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    output_root = ensure_dir(output_root)
    device = get_device(device_name)
    feature_config.validate()
    # Korpus başına öznitelik ayarı: özel ayar verilmişse onu, verilmemişse
    # genel varsayılanı kullan; hepsini tek tek doğrula.
    selected_feature_configs = {
        corpus: (feature_configs or {}).get(corpus, feature_config)
        for corpus in corpora
    }
    for corpus, selected_config in selected_feature_configs.items():
        selected_config.validate()

    results: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        results[corpus] = run_corpus(
            corpus,
            manifest_path=manifest_path,
            cache_root=cache_root,
            output_root=output_root,
            grid_mode=grid_mode,
            max_epochs=max_epochs,
            device=device,
            feature_config=selected_feature_configs[corpus],
            feature_workers=feature_workers,
            loader_workers=loader_workers,
            amp=amp,
            limit_per_split=limit_per_split,
            resume=resume,
        )

    # Karşılaştırma tablosu + tüm çalıştırmanın üst düzey özeti. Özete torch
    # ve CUDA sürümleri de yazılır: sonuçların hangi ortamda üretildiği
    # tekrarlanabilirlik açısından belgelenmiş olur.
    comparison = _write_model_comparison(results, prior_results_path, output_root)
    summary = {
        'assignment': 3,
        'manifest': str(manifest_path),
        'grid_mode': grid_mode,
        'max_epochs': max_epochs,
        'diagnostic_limit_per_split': limit_per_split,
        'resume_enabled': resume,
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'feature_config_by_dataset': {
            corpus: {
                **asdict(selected_config),
                'vector_size': selected_config.vector_size,
            }
            for corpus, selected_config in selected_feature_configs.items()
        },
        'per_dataset': results,
        'comparison_rows': comparison.to_dict(orient='records'),
    }
    _write_json(Path(output_root) / 'summary.json', summary)
    return results


def load_saved_model(
    checkpoint_path: str | Path,
    device_name: str = 'cpu',
) -> tuple[MLP, FeatureStandardizer, dict[str, Any]]:
    '''Teslim edilen bir modeli ve yalnızca-eğitimle öğrenilmiş normalizasyon parametrelerini yükler.

    best_model.pt checkpoint'inin okuma tarafı: mimariyi konfigürasyondan
    yeniden kurar, ağırlıkları yükler ve modeli değerlendirme moduna (eval)
    alır. Standardizer da checkpoint'ten geri gelir — yeni bir ses dosyasını
    sınıflandırmak için eğitimdeki DÖNÜŞÜMÜN AYNISI uygulanmak zorundadır;
    yoksa model bambaşka ölçekte girdi görür ve saçmalar.
    '''

    device = get_device(device_name)
    # weights_only=True güvenli yükleme; eski PyTorch bu parametreyi
    # tanımıyorsa (TypeError) klasik yüklemeye düş.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MLPConfig.from_dict(checkpoint['model_config'])
    model = MLP(checkpoint['input_dim'], checkpoint['num_classes'], config)
    model.load_state_dict(checkpoint['model_state_dict'])
    # eval(): dropout kapanır, BatchNorm çalışma istatistiklerini kullanır —
    # çıkarım (inference) için doğru mod.
    model.to(device).eval()
    standardizer = FeatureStandardizer(
        mean=checkpoint['standardizer_mean'].cpu().numpy(),
        scale=checkpoint['standardizer_scale'].cpu().numpy(),
    )
    return model, standardizer, checkpoint
