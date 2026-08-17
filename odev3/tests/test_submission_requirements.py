'''Derse özgü VS Code geliştirme geçmişi ve teslim dosyalarının testleri.

Bu dosya kod mantığını değil, TESLİM PAKETİNİN bütünlüğünü test eder:
- VS Code yerel geçmiş (local history) ayarları ders şartlarına uygun mu?
- README.txt her Python dosyasını gerçekten anlatıyor mu?
- Doğrulama not defteri çalıştırılmış çıktılarıyla kaydedilmiş mi?

Böylece "teslimde eksik/bayat dosya" hataları da otomasyonla yakalanır.
'''

import json
from pathlib import Path


# Testler depo köküne göre mutlak yol kurar: pytest hangi klasörden
# çalıştırılırsa çalıştırılsın aynı dosyalar bulunur.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ODEV3_ROOT = PROJECT_ROOT / 'odev3'


def test_workspace_settings_match_course_history_requirements() -> None:
    '''.vscode/settings.json, dersin istediği yerel geçmiş ayarlarını birebir içermeli.

    Bu ayarlar geliştirme sürecinin kanıtını (dosya değişiklik geçmişini)
    otomatik biriktirir; yanlışlıkla değiştirilirlerse test kırılır ve
    teslim şartı sessizce ihlal edilmemiş olur.
    '''

    settings = json.loads(
        (PROJECT_ROOT / '.vscode' / 'settings.json').read_text(encoding='utf-8')
    )

    assert settings['workbench.localHistory.enabled'] is True
    assert settings['workbench.localHistory.maxFileEntries'] == 2400
    assert settings['workbench.localHistory.maxFileSize'] == 8192
    assert settings['workbench.localHistory.mergeWindow'] == 60
    assert settings['files.autoSave'] == 'onFocusChange'


def test_plain_text_readme_explains_every_python_file() -> None:
    '''README.txt var olmalı ve HER Python dosyasının adını içermeli.

    Dosya listeleri glob ile dinamik toplanır: yeni bir .py dosyası eklenip
    README güncellenmezse bu test hemen kırılır. Ayrıca test komutu ve not
    defteri referansı da metinde aranır; README.md'nin OLMAMASI (düz metin
    şartı) ayrıca doğrulanır.
    '''

    readme_path = ODEV3_ROOT / 'README.txt'
    readme = readme_path.read_text(encoding='utf-8')
    source_files = sorted(path.name for path in ODEV3_ROOT.glob('*.py'))
    test_files = sorted(path.name for path in (ODEV3_ROOT / 'tests').glob('test_*.py'))

    assert readme_path.is_file()
    assert 'python -m pytest odev3/tests -q' in readme
    assert 'validation_test_results.ipynb' in readme
    assert all(name in readme for name in source_files)
    assert all(name in readme for name in test_files)
    assert not (ODEV3_ROOT / 'README.md').exists()


def test_validation_test_notebook_contains_saved_executed_outputs() -> None:
    '''Doğrulama not defteri ÇALIŞTIRILMIŞ ve çıktıları kaydedilmiş halde teslim edilmeli.

    Kontroller: nbformat 4, tam 6 kod hücresi, her hücrenin execution_count'u
    dolu ve çıktısı mevcut, markdown hücresi yok (yalnız kod), ve çıktı
    metinlerinde beklenen anahtar ifadeler ('DOĞRULAMA TAMAMLANDI', iki
    korpus adı, karşılaştırma alanı) geçiyor. Boş/çalıştırılmamış bir
    defter bu testten geçemez.
    '''

    notebook_path = ODEV3_ROOT / 'validation_test_results.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    code_cells = [
        cell for cell in notebook['cells'] if cell.get('cell_type') == 'code'
    ]
    # Tüm hücre çıktılarını tek metinde topla: içerik iddiaları bunun
    # üzerinde aranır.
    serialized_outputs = json.dumps(
        [cell.get('outputs', []) for cell in code_cells],
        ensure_ascii=False,
    )

    assert notebook['nbformat'] == 4
    assert len(code_cells) == 6
    assert all(cell.get('execution_count') is not None for cell in code_cells)
    assert all(cell.get('outputs') for cell in code_cells)
    assert not any(
        cell.get('cell_type') == 'markdown' for cell in notebook['cells']
    )
    assert 'DOĞRULAMA TAMAMLANDI' in serialized_outputs
    assert 'CREMA-D' in serialized_outputs
    assert 'MELD' in serialized_outputs
    assert 'absolute_difference' in serialized_outputs
