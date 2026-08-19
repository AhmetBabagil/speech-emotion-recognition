# Pitch'li tek yapılandırmayı (resmî kazanan: 32x200 BiGRU) eğitip test eder.
#
# Amaç: "Aralık serisine pitch (F0) eklemek Yöntem 2'yi iyileştiriyor mu?" sorusunu tek koşuda cevaplamak. Resmî kazananın AYNI model ayarlarını kullanır; tek fark özniteliklerin pitch içermesidir (44 -> 47). Böylece karşılaştırma adil: değişen tek şey pitch.
#
# Önce öznitelikler hazır olmalı:
# python final/hizli_cikarma.py --sadece 32 200 --islemler 14
# Sonra:
# python final/pitch_dene.py

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.dataset import Standardizer  # noqa: E402
from final.features import IntervalConfig, extract_interval_series  # noqa: E402
from final.models import OptimSettings, RNNConfig, SeqRNN, count_parameters  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import report as evaluate_report  # noqa: E402
from ser.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger(__name__)

RESMI_TEST_ACC = 0.6377     # pitch'siz resmî kazananın test doğruluğu
RESMI_TEST_F1 = 0.6440


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resmî kazananın AYNI ayarları — tek fark: öznitelikte pitch var.
    feature_cfg = IntervalConfig(n_intervals=32, interval_ms=200, use_pitch=True)
    model_cfg = RNNConfig(
        rnn_type='gru', hidden_size=192, num_layers=2, bidirectional=True,
        dropout=0.3, pooling='mean',
        optim=OptimSettings(batch_size=64, learning_rate=1e-3,
                            weight_decay=1e-4, patience=8),
    )
    log.info('Pitch denemesi: 32x200 BiGRU | öznitelik boyutu=%d (pitch %s)',
             feature_cfg.feature_dim, 'AÇIK' if feature_cfg.use_pitch else 'kapalı')

    # Resmî bölme (seed 42) — pitch'siz sonuçla adil kıyas için aynı bölme.
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    train_df, val_df, test_df = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': train_df, 'val': val_df, 'test': test_df}

    # Öznitelikler (hizli_cikarma ile önceden çıkarılmış olmalı; değilse burada
    # thread'li çıkarılır — o yüzden önce hizli_cikarma çalıştırın).
    tensorlar = _feature_folds(folds, feature_cfg, extract_interval_series,
                               'data/cache/final', workers=16, cache={})
    train_x, train_y = tensorlar['train']
    val_x, val_y = tensorlar['val']
    test_x, test_y = tensorlar['test']

    olcek = Standardizer.fit(train_x, feature_axis=2)
    model = SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)
    log.info('Model parametresi: %d', count_parameters(model))

    sonuc = train_with_early_stopping(
        model, olcek.transform(train_x), train_y,
        olcek.transform(val_x), val_y, model_cfg.optim,
        num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=42,
    )
    log.info('Geçerleme: acc=%.4f macro-F1=%.4f (en iyi epoch %d)',
             sonuc.validation_metrics['accuracy'],
             sonuc.validation_metrics['macro_f1'], sonuc.best_epoch)

    # Test değerlendirmesi.
    class_weights = inverse_frequency_weights(train_df['label_idx'].to_numpy(), NUM_CLASSES)
    out_dir = ensure_dir('final/deneyler/pitch/cremad/rnn')
    _, _, test_prob = evaluate_arrays(sonuc.model, olcek.transform(test_x), test_y,
                                      class_weights=class_weights, device=device)
    test_metrics = evaluate_report(test_y, test_prob.argmax(axis=1), out_dir,
                                   prefix='test', title='Pitch BiGRU test')

    acc, f1 = test_metrics['accuracy'], test_metrics['macro_f1']
    print('\n===== PITCH DENEMESİ SONUCU =====')
    print(f'  Pitch\'li 32x200 BiGRU : test acc={acc:.4f}  macro-F1={f1:.4f}')
    print(f'  Resmî (pitch\'siz)     : test acc={RESMI_TEST_ACC:.4f}  macro-F1={RESMI_TEST_F1:.4f}')
    fark = acc - RESMI_TEST_ACC
    if fark > 0.005:
        print(f'  -> Pitch KAZANDI (+{fark:.4f} doğruluk). Terfi düşünülebilir.')
    elif fark < -0.005:
        print(f'  -> Pitch kaybetti ({fark:.4f}). Mevcut ayarlar savunuldu.')
    else:
        print(f'  -> Fark önemsiz ({fark:+.4f}). Pitch nötr; sadelik için pitch\'siz kalınabilir.')


if __name__ == '__main__':
    main()
