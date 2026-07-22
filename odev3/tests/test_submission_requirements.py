'''Tests for course-specific VS Code development-history submission files.'''

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ODEV3_ROOT = PROJECT_ROOT / 'odev3'


def test_workspace_settings_match_course_history_requirements() -> None:
    settings = json.loads(
        (PROJECT_ROOT / '.vscode' / 'settings.json').read_text(encoding='utf-8')
    )

    assert settings['workbench.localHistory.enabled'] is True
    assert settings['workbench.localHistory.maxFileEntries'] == 2400
    assert settings['workbench.localHistory.maxFileSize'] == 8192
    assert settings['workbench.localHistory.mergeWindow'] == 60
    assert settings['files.autoSave'] == 'onFocusChange'


def test_plain_text_readme_explains_every_python_file() -> None:
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
    notebook_path = ODEV3_ROOT / 'validation_test_results.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    code_cells = [
        cell for cell in notebook['cells'] if cell.get('cell_type') == 'code'
    ]
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
