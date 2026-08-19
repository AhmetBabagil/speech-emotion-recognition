# Veri katmanı: veri indirme, manifest oluşturma, bölme ve PyTorch dataset'leri.
#
# Bu __init__, alt modüllerdeki en çok kullanılan isimleri paket seviyesine taşır; böylece diğer modüller ``from ser.data import prepare_splits`` gibi kısa importlar yazabilir ve iç dosya düzeni değişse bile dış arayüz sabit kalır.

from .dataset import SERDataset, mfcc_feature_matrix, class_weights
from .splits import prepare_splits

# __all__: "from ser.data import *" ile dışa açılan resmî isim listesi.
__all__ = ["SERDataset", "mfcc_feature_matrix", "class_weights", "prepare_splits"]
