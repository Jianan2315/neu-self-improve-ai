"""Import the deployment entrypoint with project packages strictly unavailable.

Modal is stubbed: this checks build-stage import isolation without cloud calls.
"""
from pathlib import Path
import subprocess
import sys
import unittest


class BuildImportTests(unittest.TestCase):
    def test_deployment_import_without_any_runtime_package(self):
        script = Path(__file__).resolve().parents[1] / 'search_r1_framework_modal.py'
        check = r'''
import importlib.abc
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

class BlockProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('search_r1_framework', 'search_r1_retriever', 'torch', 'verl', 'faiss')):
            raise ModuleNotFoundError('Build-stage package unavailable: ' + fullname)

sys.meta_path.insert(0, BlockProjectImports())
modal = ModuleType('modal')
modal.Image = MagicMock()
modal.Volume = MagicMock()
def decorate(*args, **kwargs):
    return lambda obj: obj
modal.App = lambda *args: SimpleNamespace(function=decorate, cls=decorate)
modal.enter = modal.exit = modal.method = modal.concurrent = decorate
sys.modules['modal'] = modal
spec = importlib.util.spec_from_file_location('isolated_deployment', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.check_retriever_dependencies)
assert callable(module.install_and_check_training_extensions)
assert module.PERFORMANCE_HARD_TIMEOUT_SECONDS == 86400
assert not any(k.startswith('search_r1_framework') for k in sys.modules)
print('BUILD_STAGE_IMPORT_ISOLATION_PASSED')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', check, str(script)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('BUILD_STAGE_IMPORT_ISOLATION_PASSED', result.stdout)


if __name__ == '__main__':
    unittest.main()
