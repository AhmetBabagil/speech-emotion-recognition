'''Resmi Yöntem 2 modelini (jitter+kontrast, 53 boyut) 5 koşu eğitip kaydeder.

Metodoloji (dürüstlük için önemli):
  * Manşet sayı = 5 koşunun ORTALAMASI (tek bir şanslı koşu değil).
  * Kaydedilen model (improved_model.pt) = 5 koşu içinde GEÇERLEMEDE (validation)
    en iyi olan koşu. Modeli test setine bakarak seçmek test sızıntısıdır; bu yüzden
    seçimi yalnızca geçerleme macro-F1'ine göre yapıyoruz.
  * Kaydedilen modelin karışıklık matrisi ve öğrenme eğrisi, sunumun beklediği
    standart isimlerle (test_confusion_matrix.png, winner_learning_curve.png)
    yeniden çizilir; böylece slaytlar güncel modeli gösterir.
'''

from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.ablasyon import MODEL, on_cikar_paralel  # noqa: E402
from final.dataset import Standardizer  # noqa: E402
from final.features import IntervalConfig, extract_interval_series  # noqa: E402
from final.models import SeqRNN  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds, _plot_history  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import compute_metrics, report as evaluate_report  # noqa: E402


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcfg = IntervalConfig(n_intervals=32, interval_ms=200)  # varsayilan = jitter+kontrast (53)
    print(f'Resmi Yontem 2 oznitelik boyutu: {fcfg.feature_dim}')

    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),
                     dict(fcfg.__dict__), 'data/cache/final', 14)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',
                       workers=8, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=2)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

    # 5 koşu: her koşunun test metriklerini VE geçerleme macro-F1'ini topla.
    accs, bals, f1s, kosular = [], [], [], []
    for k in range(5):
        torch.manual_seed(k)
        torch.cuda.manual_seed_all(k)
        model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)
        res = train_with_early_stopping(model, tx_s, ty, vx_s, vy, MODEL.optim,
                                        num_classes=NUM_CLASSES, device=device,
                                        max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)
        m = compute_metrics(ey, prob.argmax(axis=1))
        accs.append(m['accuracy'])
        bals.append(m['balanced_accuracy'])
        f1s.append(m['macro_f1'])
        val_f1 = float(res.validation_metrics['macro_f1'])
        kosular.append({
            'val_f1': val_f1,
            'state': {n: v.detach().cpu().clone() for n, v in res.model.state_dict().items()},
            'history': res.history,
            'test_acc': m['accuracy'],
        })
        print(f'  kosu {k}: val_macroF1={val_f1:.4f}  test_acc={m["accuracy"]:.4f}')

    # Modeli GEÇERLEMEDE en iyi koşuya göre seç (test setine bakmadan).
    secilen = max(kosular, key=lambda r: r['val_f1'])

    out = Path('final/outputs/cremad/rnn')
    model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)
    model.load_state_dict(secilen['state'])
    model.to(device).eval()
    _, _, prob = evaluate_arrays(model, ex_s, ey, class_weights=cw, device=device)
    # Standart isimlerle yaz: sunum bu dosyalari okuyor -> guncel modeli gosterir.
    tm = evaluate_report(ey, prob.argmax(axis=1), out, prefix='test',
                         title='Yontem 2 (BiGRU) — jitter+kontrast (53) test')
    _plot_history(secilen['history'], out / 'winner_learning_curve.png',
                  'Yontem 2 (BiGRU) — ogrenme egrisi')
    torch.save({'state_dict': model.state_dict(), 'variant': 'jitter_contrast53',
                'feature_config': fcfg.__dict__, 'model_config': MODEL.to_dict(),
                'standardizer_mean': olcek.mean, 'standardizer_scale': olcek.scale,
                'feature_axis': olcek.feature_axis}, out / 'improved_model.pt')

    ortalama = {
        'variant': 'jitter+contrast (53)', 'feature_dim': fcfg.feature_dim, 'runs': 5,
        'selection': 'validation macro-F1 (test sizintisi yok)',
        'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
        'mean_balanced_accuracy': float(np.mean(bals)),
        'mean_macro_f1': float(np.mean(f1s)), 'std_macro_f1': float(np.std(f1s)),
        'best_run_accuracy': float(np.max(accs)),
        'baseline_test_accuracy': 0.6455,   # taban (44 boyut) 5-kosu ort.
        'selected_model_test': tm,
    }
    (out / 'pitch_summary.json').write_text(
        json.dumps(ortalama, indent=2, default=str), encoding='utf-8')

    print(f'\n5-kosu ortalama: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  '
          f'bal={np.mean(bals):.4f}  macro-F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}')
    print(f'Kaydedilen model (gecerlemeye gore secildi): test_acc={tm["accuracy"]:.4f}  '
          f'macro-F1={tm["macro_f1"]:.4f}')
    print(f'disgust F1: {tm["per_class"]["disgust"]["f1"]:.3f}')


if __name__ == '__main__':
    main()
