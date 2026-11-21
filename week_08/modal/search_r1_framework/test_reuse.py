"""Local checks for rerunning setup without downloads, overwrite or initialization."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from search_r1_framework.config import MODEL_ID, Settings
from search_r1_framework.data import prepare
from search_r1_framework.workbook_upload import ensure_workbook


class ReuseTests(unittest.TestCase):
    def test_workbook_same_missing_different_and_network_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'book.xlsx')
            p.write_bytes(b'benchmark')
            volume = MagicMock()
            volume.read_file.return_value = iter([b'bench', b'mark'])
            self.assertEqual(ensure_workbook(volume, p, '/book.xlsx'), 'reused')
            volume.batch_upload.assert_not_called()
            volume.read_file.return_value = iter([b'different'])
            with self.assertRaises(ValueError):
                ensure_workbook(volume, p, '/book.xlsx')
            volume.batch_upload.assert_not_called()
            volume.read_file.side_effect = ConnectionError('offline')
            with self.assertRaises(ConnectionError):
                ensure_workbook(volume, p, '/book.xlsx')
            volume.batch_upload.assert_not_called()
            volume.read_file.side_effect = FileNotFoundError()
            self.assertEqual(ensure_workbook(volume, p, '/book.xlsx'), 'uploaded')
            volume.batch_upload.return_value.__enter__.return_value.put_file.assert_called_once_with(str(p), '/book.xlsx')

    def test_reuse_preparation_validates_files_settings_and_adapters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            workbook = root / 'book.xlsx'
            workbook.write_bytes(b'benchmark')
            target = root / 'prepared'
            target.mkdir()
            model = root / 'model'
            model.mkdir()
            settings = Settings()
            manifest = {'model_id': MODEL_ID, 'model_path': str(model), 'settings': settings.to_dict(),
                        'benchmark_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(), 'files_sha256': {}}
            for name in ('train.parquet', 'dev.parquet', 'bamboogle_direct.parquet', 'bamboogle_search.parquet'):
                (target / name).write_bytes(name.encode())
                manifest['files_sha256'][name] = hashlib.sha256(name.encode()).hexdigest()
            for name in ('initial_actor', 'initial_critic'):
                (target / name).mkdir()
                for file in ('adapter_config.json', 'adapter_model.safetensors'):
                    (target / name / file).write_bytes(b'test')
            (target / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(prepare(target, workbook, settings, reuse_existing=True), manifest)
            with self.assertRaises(ValueError):
                prepare(target, workbook, replace(settings, seed=7), reuse_existing=True)
            with self.assertRaises(FileExistsError):
                prepare(target, workbook, settings)
            (target / 'initial_actor' / 'adapter_model.safetensors').unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(target, workbook, settings, reuse_existing=True)

    def test_both_notebooks_call_the_shared_upload_helper(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import sys
        root = Path(__file__).resolve().parents[1]
        for name in ('assignment_08_tests.ipynb', 'assignment_08_tests_zh.ipynb'):
            cells = json.loads((root / name).read_text(encoding='utf-8'))['cells']
            source = next(''.join(c['source']) for c in cells if c['cell_type'] == 'code'
                          and ''.join(c['source']).startswith('UPLOAD_WORKBOOK ='))
            volume = object()
            modal = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *args: volume))
            with patch.dict(sys.modules, {'modal': modal}), patch('builtins.print'), \
                 patch('search_r1_framework.workbook_upload.ensure_workbook', return_value='reused') as ensure:
                exec(source, {})
                ensure.assert_called_once_with(volume, 'Bamboogle Prerelease.xlsx', '/Bamboogle Prerelease.xlsx')


if __name__ == '__main__':
    unittest.main()
