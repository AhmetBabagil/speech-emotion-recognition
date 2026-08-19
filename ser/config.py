# Deney yapılandırması: YAML'dan yüklenip YAML'a yazılabilen iç içe dataclass'lar.
#
# Neden böyle bir tasarım? Tek bir ``Config`` nesnesi; veri yükleme, öznitelik çıkarımı, model kurulumu, eğitim ve değerlendirme aşamalarının HEPSİNE parametre olarak taşınır. Böylece her aşama ses parametrelerinde (örnekleme frekansı, klip uzunluğu, mel bant sayısı, ...) hemfikir olur. Alternatif olan "her modül kendi sabitini tanımlasın" yaklaşımı, bir yerde 16 kHz diğerinde 22 kHz kullanılması gibi sessiz uyumsuzluklara yol açardı.
#
# Dataclass kullanmanın avantajları:
# * Varsayılan değerler kodda, tek bakışta görülür ve tip ipuçlarıyla belgelidir.
# * ``asdict`` ile YAML'a kolayca dökülür; her deneyin yanına config.yaml
# kaydedilerek deney tekrarlanabilirliği (reproducibility) sağlanır.
# * YAML'daki bir bölüm eksikse varsayılanlar devreye girer; deney dosyaları
# yalnızca değiştirmek istedikleri alanları yazar.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .constants import SAMPLE_RATE


@dataclass
class AudioConfig:  # Ham ses ile ilgili ayarlar: örnekleme frekansı ve sabit klip süresi.

    sample_rate: int = SAMPLE_RATE
    # Spektrogram/dalga formu batch'lemek için kullanılan SABİT klip uzunluğu
    # (saniye). Derin ağlar aynı boyutlu girdiler ister; kayıtların süresi ise
    # değişkendir. Çözüm: kısa klipler sıfırla doldurulur (padding), uzun
    # klipler kırpılır (eğitimde rastgele konumdan — bu ufak bir veri
    # çoğaltma etkisi de yaratır; değerlendirmede ortadan — deterministik).
    clip_seconds: float = 4.0

    @property
    def num_samples(self) -> int:
        # Saniye cinsinden süreyi örnek (sample) sayısına çevirir:
        # 16000 Hz * 4.0 s = 64000 örnek. round() olası float hatalarını giderir.
        return int(round(self.sample_rate * self.clip_seconds))


@dataclass
class FeatureConfig:  # Öznitelik çıkarımı (STFT/mel/MFCC) ve spektrogram augmentasyon ayarları.

    # Ortak STFT parametreleri (hem MFCC hem mel-spektrogram bunları kullanır).
    # n_fft=1024 @16kHz => 64 ms'lik pencere: frekans çözünürlüğü ile zaman
    # çözünürlüğü arasında konuşma için dengeli bir seçim.
    n_fft: int = 1024
    # hop_length=256 => pencereler 16 ms'de bir kayar; ardışık kareler %75 örtüşür.
    hop_length: int = 256
    win_length: int = 1024
    # Mel filtre bankasındaki bant sayısı: spektrogramın "yükseklik" ekseni.
    n_mels: int = 64
    # İlgilenilen frekans aralığı. 20 Hz altı gürültü/DC'dir; 8 kHz üstü konuşma
    # için bilgi taşımaz (zaten 16 kHz örnekleme ile Nyquist sınırı 8 kHz'dir).
    fmin: float = 20.0
    fmax: float | None = 8000.0  # <= sample_rate / 2 olmalı (Nyquist)
    # MFCC (klasik taban modeli) parametresi: kaç katsayı alınacağı.
    n_mfcc: int = 40
    # SpecAugment tarzı maskeleme (yalnızca EĞİTİM sırasında log-mel üzerine
    # uygulanır): spektrogramda rastgele bir frekans bandı ve bir zaman dilimi
    # sıfırlanır. Model tek bir banda/ana bağımlı kalamaz, ezber azalır.
    freq_mask: int = 8   # maskelenebilecek en fazla mel bandı sayısı
    time_mask: int = 16  # maskelenebilecek en fazla zaman karesi sayısı
    augment: bool = True


@dataclass
class ModelConfig:  # Hangi modelin kurulacağı ve modele özgü hiperparametreler.

    # Seçenekler: "cnn" veya "wav2vec2". (MFCC taban modeli sklearn tabanlıdır
    # ve ayrı bir yoldan kurulur; bu alanı dikkate almaz.)
    name: str = "cnn"
    # Dropout: aşırı öğrenmeye (overfitting) karşı, eğitimde nöronların bir
    # kısmını rastgele kapatan düzenlileştirme tekniği.
    dropout: float = 0.3
    # CNN'e özgü: her konvolüsyon bloğunun kanal sayısı. Derine indikçe kanal
    # sayısını ikiye katlamak (32→64→128→256) klasik bir desendir: uzamsal
    # çözünürlük düşerken temsil zenginliği artar.
    cnn_channels: tuple[int, ...] = (32, 64, 128, 256)
    # wav2vec2'ye özgü: Hugging Face üzerindeki önceden eğitilmiş model adı ve
    # konvolüsyonel öznitelik kodlayıcısının dondurulup dondurulmayacağı.
    # Küçük SER veri kümelerinde kodlayıcıyı dondurmak overfitting'i azaltır.
    pretrained_name: str = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True


@dataclass
class TrainConfig:  # Eğitim döngüsünün hiperparametreleri.

    epochs: int = 50
    batch_size: int = 32
    # AdamW için öğrenme hızı; 3e-4 küçük CNN'ler için sağlam bir başlangıçtır.
    lr: float = 3e-4
    # L2 tarzı ağırlık cezası (AdamW'nin "decoupled" weight decay'i).
    weight_decay: float = 1e-4
    # CrossEntropyLoss için sınıf ağırlıklandırma stratejisi:
    # "none" | "inverse" | "balanced". MELD'de sınıflar çok dengesiz olduğundan
    # (neutral çok, fear/disgust az) azınlık sınıflara daha yüksek ağırlık
    # vermek makro-F1'i belirgin iyileştirir.
    class_weighting: str = "balanced"
    # Etiket yumuşatma: modelin %100 emin olmasını cezalandırır; SER etiketleri
    # zaten öznel/gürültülü olduğu için küçük bir değer iyi gelir.
    label_smoothing: float = 0.05
    # Doğrulama metriği bu kadar epoch boyunca iyileşmezse eğitim erken durur.
    early_stop_patience: int = 8
    # DataLoader işçi süreci sayısı. 0 = ana süreçte yükle; Windows'ta çoklu
    # süreç (fork yerine spawn) sorun çıkarabildiği için güvenli varsayılandır.
    # Linux/GPU makinede artırılabilir.
    num_workers: int = 0
    # Gradyan patlamasına karşı norm kırpma eşiği (0 => kapalı).
    grad_clip: float = 5.0
    # Tüm rastgelelik kaynakları için tohum; deneyin tekrarlanabilirliği için.
    seed: int = 42
    # Doğrulamada izlenecek metrik: "macro_f1" veya "accuracy". Dengesiz veri
    # yüzünden makro-F1 tercih edilir (her sınıfa eşit önem verir).
    monitor: str = "macro_f1"
    # Otomatik karışık hassasiyet (float16): GPU'da eğitimi hızlandırır ve
    # bellek kazandırır; CPU'da sessizce yok sayılır.
    amp: bool = True


@dataclass
class DataConfig:  # Verinin nereden okunacağı ve nasıl bölüneceği.

    # Tüm kullanılabilir kayıtları listeleyen birleşik CSV (build_manifest üretir).
    manifest: str = "data/processed/manifest.csv"
    # Bu koşuda hangi korpuslar kullanılacak? Örn. ["cremad"], ["meld"] ya da
    # ikisi birden. train != eval ise otomatik olarak cross-corpus moduna girilir.
    train_corpora: tuple[str, ...] = ("cremad",)
    eval_corpora: tuple[str, ...] = ("cremad",)
    # Bölme stratejisi (yalnızca korpus-içi modda geçerli):
    #   "speaker"       -> konuşmacı-bağımsız (bir konuşmacının tüm kayıtları tek
    #                      folda gider; SER için DOĞRU protokol, çünkü model
    #                      duyguyu değil sesi/kimliği ezberleyebilir),
    #   "meld_official" -> MELD'in resmî diyalog-bazlı foldları (konuşmacı-
    #                      bağımsız DEĞİLDİR ama literatürle karşılaştırılabilir),
    #   "random"        -> tamamen rastgele satır bölmesi (yalnızca kıyas için).
    # Cross-corpus modda bu alan yok sayılır: eğitim korpusu her zaman konuşmacı-
    # bağımsız şekilde train/val'e bölünür, eval korpusunun TAMAMI test olur.
    split: str = "speaker"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # Öznitelikler diske .npy olarak önbelleklensin mi? Spektrogram çıkarımı
    # pahalıdır; ikinci epoch'tan itibaren büyük hız kazancı sağlar.
    cache_features: bool = True
    cache_dir: str = "data/cache"


@dataclass
class Config:
    # Tüm alt yapılandırmaları bir araya getiren kök nesne.
    #
    # ``field(default_factory=...)`` kullanılır çünkü dataclass'larda değişebilir (mutable) varsayılanlar doğrudan yazılamaz — her Config örneği kendi alt nesnelerini almalıdır, yoksa iki deney aynı AudioConfig'i paylaşırdı.

    experiment: str = "default"   # çıktı klasörünün adı bu isimden türetilir
    output_dir: str = "outputs"
    audio: AudioConfig = field(default_factory=AudioConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # ---- Serileştirme / geri yükleme ------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        # İç içe sözlükten Config kurar (YAML yüklemesinin arka planı).
        #
        # Her alt bölüm kendi dataclass'ına çevrilir; tanınmayan anahtarlar sessizce atlanır ki YAML'a eklenen bir not/açıklama yüklemeyi bozmasın.
        d = dict(d or {})  # kopya al: çağıranın sözlüğünü değiştirmeyelim
        sub = {
            "audio": AudioConfig,
            "feature": FeatureConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "data": DataConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            # pop: alt bölümü sözlükten çıkar; kalanlar üst düzey alanlardır.
            # "or {}": YAML'da bölüm boş bırakılırsa None gelir, onu da tolere et.
            section = d.pop(key, {}) or {}
            kwargs[key] = _build_dataclass(klass, section)
        # Geriye kalan üst düzey skaler alanlar (experiment, output_dir);
        # bilinmeyen anahtarlar yok sayılır, böylece YAML dosyasındaki fazladan
        # bir açıklama alanı yüklemeyi çökertmez.
        valid = {f.name for f in dataclasses.fields(cls)}
        for k, v in d.items():
            if k in valid:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        # Bir YAML dosyasından Config yükler.
        #
        # ``safe_load`` kullanılır: YAML içinde rastgele Python nesnesi kurmaya izin vermez (güvenlik). Boş dosya None döndürür; "or {}" bunu karşılar.
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        # asdict, iç içe dataclass'ları da özyinelemeli olarak sözlüğe çevirir.
        return asdict(self)

    def save(self, path: str | Path) -> None:
        # Config'i YAML olarak diske yazar (deney kaydı/tekrarlanabilirlik).
        #
        # ``sort_keys=False``: alanlar tanım sırasında kalsın; okunması kolay olur.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _build_dataclass(klass, section: dict[str, Any]):
    # Sözlükten bir dataclass kurar; iki pürüzü giderir:
    #
    # 1. YAML listeleri Python list olarak gelir ama bazı alanlar tuple bekler
    # (örn. cnn_channels). Tuple tipli alanlar için list → tuple dönüşümü yapılır.
    # 2. Tanınmayan anahtarlar yok sayılır. (Yazım hataları için açık bir hata
    # fırlatmak daha "temiz" olurdu; ama config dosyalarının ek not/annotation
    # taşıyabilmesi için bilinçli olarak hoşgörülü bırakıldı.)
    valid = {f.name for f in dataclasses.fields(klass)}
    # Tip metnine bakarak tuple isteyen alanları bul ("tuple[int, ...]" gibi).
    # `from __future__ import annotations` yüzünden f.type bir dizgedir; bu
    # nedenle isinstance yerine dizge içinde "tuple" aranır.
    tuple_fields = {
        f.name for f in dataclasses.fields(klass)
        if "tuple" in str(f.type).lower()
    }
    kwargs = {}
    for k, v in (section or {}).items():
        if k not in valid:
            continue  # bilinmeyen anahtar: sessizce atla
        if k in tuple_fields and isinstance(v, list):
            v = tuple(v)  # YAML listesi -> tuple (alan tipiyle uyum için)
        kwargs[k] = v
    return klass(**kwargs)
