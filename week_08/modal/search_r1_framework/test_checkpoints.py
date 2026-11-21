"""Recovery contracts using small synthetic checkpoint archives on local CPU."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from search_r1_framework.checkpoints import (required_files, seal_latest_checkpoint,
    select_checkpoint, validate_resume_request, activate_recovery)
from search_r1_framework.config import Settings, overrides


def make_checkpoint(root, algorithm, step):
    folder = root / "checkpoints" / f"global_step_{step}"
    for name in required_files(algorithm):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pt":
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", b"synthetic local test")
        elif path.suffix == ".json":
            path.write_text(json.dumps({"world_size": 1}))
        else:
            path.write_bytes(b"synthetic weights")
    (root / "checkpoints/latest_checkpointed_iteration.txt").write_text(str(step))
    return folder


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.info = {"job": "train_ppo", "settings": replace(Settings(), steps=2).to_dict(),
                     "preparation_name": "test", "data_manifest": {"revision": "fixed"},
                     "smoke_dev_questions": 1}
        (self.root / "manifest.json").write_text(json.dumps(self.info))
        (self.root / "runtime_config").mkdir()
        for name in ("tools.yaml", "agents.yaml"):
            (self.root / "runtime_config" / name).write_text("{}")

    def test_both_algorithms_restore_full_state_and_explicit_path(self):
        for algorithm in ("ppo", "grpo"):
            with self.subTest(algorithm=algorithm):
                make_checkpoint(self.root, algorithm, 1)
                receipt = self.root / "checkpoint_receipts/1.json"
                if receipt.exists():
                    receipt.unlink()
                seal_latest_checkpoint(self.root, algorithm)
                result = select_checkpoint(self.root, algorithm, 2)
                self.assertEqual(result["remaining_steps"], 1)
                values = dict(a.split("=", 1) for a in overrides(
                    Settings(), algorithm, "/data", "/out", "/model", "/actor", "/critic",
                    resume=True, resume_checkpoint=result["path"]))
                self.assertEqual(json.loads(values["trainer.resume_mode"]), "resume_path")
                self.assertEqual(json.loads(values["trainer.resume_from_path"]), result["path"])

    def test_missing_receipt_or_path_never_starts_from_scratch(self):
        make_checkpoint(self.root, "ppo", 1)
        with self.assertRaisesRegex(ValueError, "Refusing to start from scratch"):
            select_checkpoint(self.root, "ppo", 2)
        with self.assertRaisesRegex(ValueError, "explicitly verified"):
            overrides(Settings(), "ppo", "/data", "/out", "/model", "/a", "/c", resume=True)

    def test_missing_critic_optimizer_and_truncated_archive_rejected(self):
        folder = make_checkpoint(self.root, "ppo", 1)
        (folder / "critic/optim_world_size_1_rank_0.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            seal_latest_checkpoint(self.root, "ppo")
        make_checkpoint(self.root, "ppo", 1)
        (folder / "data.pt").write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "Incomplete PyTorch"):
            seal_latest_checkpoint(self.root, "ppo")

    def test_corrupt_latest_falls_back_and_archives_rejected_receipt(self):
        make_checkpoint(self.root, "ppo", 1)
        seal_latest_checkpoint(self.root, "ppo")
        latest = make_checkpoint(self.root, "ppo", 2)
        seal_latest_checkpoint(self.root, "ppo")
        # Same-sized corruption must also be detected, not just a missing file.
        path = latest / "actor/lora_adapter/adapter_model.safetensors"
        path.write_bytes(b"x" * path.stat().st_size)
        result = select_checkpoint(self.root, "ppo", 2)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["rejected_newer_checkpoints"][0]["step"], 2)
        activate_recovery(self.root, result)
        self.assertFalse((self.root / "checkpoint_receipts/2.json").exists())
        self.assertTrue(list((self.root / "rejected_checkpoint_receipts").glob("*/2.json")))
        make_checkpoint(self.root, "ppo", 2)
        seal_latest_checkpoint(self.root, "ppo")
        self.assertEqual(select_checkpoint(self.root, "ppo", 2)["step"], 2)

    def test_settings_finished_run_and_missing_dataloader_fail(self):
        folder = make_checkpoint(self.root, "ppo", 1)
        seal_latest_checkpoint(self.root, "ppo")
        self.assertEqual(validate_resume_request(self.root, self.info)["step"], 1)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_resume_request(self.root, {**self.info, "settings": {}})
        (self.root / "finished.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "already finished"):
            validate_resume_request(self.root, self.info)
        (self.root / "finished.json").unlink()
        (folder / "data.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Refusing to start from scratch"):
            validate_resume_request(self.root, self.info)

    def test_final_checkpoint_needs_only_artifact_publication(self):
        make_checkpoint(self.root, "ppo", 2)
        seal_latest_checkpoint(self.root, "ppo")
        self.assertEqual(validate_resume_request(self.root, self.info)["remaining_steps"], 0)

    def test_interruption_test_really_terminates_child(self):
        from search_r1_framework.runner import run_framework, CheckpointTestInterrupted
        original = subprocess.Popen
        children = []
        def launch(*args, **kwargs):
            marker = self.root / "checkpoints/latest_checkpointed_iteration.txt"
            code = ("from pathlib import Path; import time; "
                    f"p=Path({str(marker)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
                    "p.write_text('1'); print('saved', flush=True); time.sleep(30)")
            child = original([sys.executable, "-c", code], **kwargs)
            children.append(child)
            return child
        progress = unittest.mock.Mock()
        with (patch('search_r1_framework.runner.subprocess.run', return_value=SimpleNamespace(stdout='')),
              patch('search_r1_framework.runner.subprocess.Popen', side_effect=launch)):
            with self.assertRaises(CheckpointTestInterrupted):
                run_framework([], self.root, progress=progress, interrupt_after_checkpoint=1)
        self.assertIsNotNone(children[0].poll())
        self.assertGreaterEqual(progress.call_count, 1)


if __name__ == "__main__":
    unittest.main()
