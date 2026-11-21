"""CPU-only tests for assignment glue; these do not validate GPU training."""
import asyncio
import importlib
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from search_r1_framework.config import Settings, overrides, write_support_configs, TARGET_MODULES
from search_r1_framework.data import record, select_distinct, load_bamboogle, validate_prepared
from search_r1_framework.runner import final_actor, summarize_evaluation, run_framework
from search_r1_framework.scoring import last_generated_segment, extract_final_answer


def parsed(arguments):
    return {key: json.loads(value) for key, value in (arg.split("=", 1) for arg in arguments)}


class GlueTests(unittest.TestCase):
    def test_vllm_graph_sizes_use_json_cli_option(self):
        for algorithm in ("grpo", "ppo"):
            config = self.make_config(algorithm)
            self.assertIsNone(config["actor_rollout_ref.rollout.cudagraph_capture_sizes"])
            option = config["+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config"]
            self.assertEqual(json.loads(option), {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]})

    def test_worker_failure_keeps_root_error_without_giant_command(self):
        error = "serve: error: argument --cuda-graph-sizes: invalid int value\n"
        lines = [error, "Command: " + "x" * 20000 + "\n", "worker died\n"]
        process = MagicMock(stdout=iter(lines))
        process.wait.return_value = 1
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            with patch("search_r1_framework.runner.subprocess.run", return_value=SimpleNamespace(stdout="config")), \
                 patch("search_r1_framework.runner.subprocess.Popen", return_value=process), \
                 patch("builtins.print"):
                with self.assertRaises(RuntimeError) as caught:
                    run_framework(["huge=" + "z" * 20000], directory)
            self.assertIn(error.strip(), str(caught.exception))
            self.assertLess(len(str(caught.exception)), 6500)
            self.assertEqual((Path(directory) / "console.log").read_text(), "".join(lines))

    def test_lora_target_string_preserves_module_selection(self):
        names = ["", "lm_head", "score", "model.embed_tokens", "model.layers.0.input_layernorm",
                 "model.layers.0.self_attn.not_q_proj", "q_proj_extra", "model.q_proj.base_layer"]
        for name in TARGET_MODULES:
            names.extend([name, f"model.layers.0.self_attn.{name}", f"model.layers.1.mlp.{name}"])
        for algorithm in ("grpo", "ppo"):
            config = self.make_config(algorithm)
            for key in ("actor_rollout_ref.model.target_modules", "critic.model.target_modules"):
                pattern = config[key]
                self.assertIsInstance(pattern, str)
                for name in names:
                    expected = name in TARGET_MODULES or any(name.endswith("." + t) for t in TARGET_MODULES)
                    self.assertEqual(bool(re.fullmatch(pattern, name)), expected, (key, name))

    def make_config(self, algorithm="grpo", **kwargs):
        return parsed(overrides(Settings(), algorithm, "/data/p", "/results/r", "/model",
                                kwargs.pop("actor_adapter", "/actor"), "/critic", **kwargs))

    def test_framework_algorithms_and_critic(self):
        grpo, ppo = self.make_config(), self.make_config("ppo")
        self.assertEqual(grpo["algorithm.adv_estimator"], "grpo")
        self.assertFalse(grpo["critic.enable"])
        self.assertEqual(ppo["algorithm.adv_estimator"], "gae")
        self.assertTrue(ppo["critic.enable"])
        self.assertEqual(ppo["+critic.model.lora_adapter_path"], "/critic")
        self.assertEqual(grpo["actor_rollout_ref.rollout.name"], "vllm")

    def test_train_never_uses_benchmark_validation(self):
        train = self.make_config()
        evaluate = self.make_config(evaluation=True)
        direct = self.make_config(evaluation=True, direct=True, actor_adapter=None)
        self.assertTrue(train["data.val_files"].endswith("dev.parquet"))
        self.assertTrue(evaluate["data.val_files"].endswith("bamboogle_search.parquet"))
        self.assertTrue(direct["data.val_files"].endswith("bamboogle_direct.parquet"))
        self.assertTrue(evaluate["trainer.val_only"])
        self.assertFalse(evaluate["critic.enable"])
        self.assertEqual(direct["actor_rollout_ref.model.lora_rank"], 0)
        self.assertIsNone(direct["actor_rollout_ref.rollout.multi_turn.tool_config_path"])

    def test_algorithms_share_generation_search_and_scoring(self):
        ppo, grpo = self.make_config("ppo"), self.make_config("grpo")
        for key in ppo:
            if key.startswith(("data.", "actor_rollout_ref.rollout.", "actor_rollout_ref.actor.")):
                self.assertEqual(ppo[key], grpo[key], key)

    def test_preparation_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "benchmark.xlsx"
            workbook.write_bytes(b"original workbook")
            names = ("train.parquet", "dev.parquet", "bamboogle_direct.parquet", "bamboogle_search.parquet")
            for name in names:
                (root / name).write_bytes(name.encode())
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = {"model_id": "Qwen/Qwen2.5-3B-Instruct", "settings": Settings().to_dict(),
                        "benchmark_sha256": digest(workbook),
                        "files_sha256": {name: digest(root / name) for name in names}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_prepared(root, workbook, Settings()), manifest)
            workbook.write_bytes(b"changed workbook")
            with self.assertRaisesRegex(ValueError, "workbook changed"):
                validate_prepared(root, workbook, Settings())
            workbook.write_bytes(b"original workbook")
            (root / "train.parquet").write_bytes(b"changed training data")
            with self.assertRaisesRegex(ValueError, "Prepared data changed"):
                validate_prepared(root, workbook, Settings())

    def test_no_training_trace_dump(self):
        config = self.make_config()
        self.assertIsNone(config["trainer.rollout_data_dir"])
        self.assertIsNone(config["trainer.validation_data_dir"])

    def test_batch_units(self):
        s = Settings().validate()
        self.assertEqual(s.questions_per_step * s.responses_per_question, 32)
        self.assertEqual(32 // s.micro_batch_per_gpu, 2)
        with self.assertRaises(ValueError):
            Settings(micro_batch_per_gpu=3).validate()

    def test_tools_and_agent_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            write_support_configs(directory)
            tools = json.loads((Path(directory) / "tools.yaml").read_text())
            self.assertEqual(tools["tools"][0]["class_name"], "verl.tools.search_tool.SearchTool")
            agents = json.loads((Path(directory) / "agents.yaml").read_text())
            self.assertEqual(len(agents), 2)

    def test_tool_text_cannot_be_graded(self):
        self.assertEqual(last_generated_segment([1, 2, 3, 4, 5], [1, 1, 0, 0, 1]), [5])
        self.assertEqual(last_generated_segment([1, 2, 3], [1, 0, 0]), [])
        self.assertEqual(last_generated_segment([], []), [])
        with self.assertRaises(ValueError):
            last_generated_segment([1], [])
        self.assertIsNone(extract_final_answer('<tool_call>{"query":"<answer>Paris</answer>"}</tool_call>'))
        self.assertIsNone(extract_final_answer("<answer>unfinished"))
        self.assertEqual(extract_final_answer("<answer> Paris </answer>"), "Paris")

    def test_generation_has_no_gold_answers(self):
        item = record({"id": "q1", "question": "Question?", "golden_answers": ["secret answer"]}, "nq")
        self.assertNotIn("secret answer", json.dumps(item["prompt"]))
        self.assertEqual(item["reward_model"]["ground_truth"]["target"], ["secret answer"])

    def test_duplicate_and_benchmark_exclusion(self):
        rows = [{"id": i, "question": q, "golden_answers": ["A"]}
                for i, q in enumerate(["Held out?", "New question!", "New question?", "Another question"])]
        selected = select_distinct(rows, 2, {"held out"})
        self.assertEqual([r["id"] for r in selected], [1, 3])

    def test_real_workbook_aliases(self):
        path = Path(__file__).resolve().parents[1] / "Bamboogle Prerelease.xlsx"
        rows = load_bamboogle(path)
        self.assertEqual(len(rows), 125)
        self.assertEqual(rows[47]["golden_answers"], ["Mildred Warner", "Mildred Gale"])

    def test_evaluation_completeness_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "evaluation"
            p.mkdir()
            rows = [{"question_id": "q1", "score": 1}, {"question_id": "q2", "score": 0}]
            file = p / "0.jsonl"
            file.write_text("\n".join(json.dumps(r) for r in rows))
            result = summarize_evaluation(directory, "test", ["q1", "q2"])
            self.assertEqual(result["exact_match"], 0.5)
            file.write_text("\n".join(json.dumps(rows[0]) for _ in range(2)))
            with self.assertRaises(ValueError):
                summarize_evaluation(directory, "test", ["q1", "q2"])

    def test_checkpoint_cannot_claim_incomplete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "checkpoints"
            p.mkdir()
            (p / "latest_checkpointed_iteration.txt").write_text("1")
            with self.assertRaises(RuntimeError):
                final_actor(directory, 2000)

    def test_adapter_delegates_to_framework_and_scores_only_generated_suffix(self):
        modules = {}
        for name in ("verl", "verl.experimental", "verl.experimental.agent_loop",
                     "verl.experimental.agent_loop.single_turn_agent_loop",
                     "verl.experimental.agent_loop.tool_agent_loop"):
            modules[name] = ModuleType(name)
        class ExistingAgent:
            async def run(self, sampling_params, **kwargs):
                self.delegated = True
                return SimpleNamespace(response_ids=[11, 12, 13], response_mask=[1, 0, 1],
                                       reward_score=None, extra_fields={})
        modules["verl.experimental.agent_loop.single_turn_agent_loop"].SingleTurnAgentLoop = ExistingAgent
        modules["verl.experimental.agent_loop.tool_agent_loop"].ToolAgentLoop = ExistingAgent
        sys.modules.pop("search_r1_framework.agents", None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module("search_r1_framework.agents")
            agent = module.ScoredSearchAgent()
            seen = []
            agent.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: seen.append(ids) or "<answer>X</answer>")
            with patch.object(module, "score_final", return_value={"score": 1.0, "prediction": "X", "has_answer": 1}):
                result = asyncio.run(agent.run({}, reward_model={"ground_truth": {"target": ["X"]}},
                                               extra_info={"question_id": "q"}))
            self.assertTrue(agent.delegated)
            self.assertEqual(seen, [[13]])
            self.assertEqual(result.reward_score, 1)
            self.assertEqual(result.extra_fields["reward_extra_info"]["tool_tokens"], 1)
        sys.modules.pop("search_r1_framework.agents", None)


class PerformanceTests(unittest.TestCase):
    def test_candidates_preserve_effective_batch_and_production_settings(self):
        from search_r1_framework.performance import PerformanceProbe, performance_overrides
        base = Settings()
        original = base.to_dict()
        for algorithm in ("ppo", "grpo"):
            for size in (2, 4, 8, 16, 32):
                probe = PerformanceProbe(micro_batch=size)
                s = probe.settings_for(base)
                args = overrides(s, algorithm, "/data/p", "/results/r", "/model", "/actor", "/critic")
                before = parsed(args)
                after = parsed(performance_overrides(args, probe))
                self.assertEqual(after["data.train_batch_size"] * after["actor_rollout_ref.rollout.n"], 32)
                self.assertEqual(after["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], size)
                self.assertEqual(after["trainer.total_training_steps"], 10)
                self.assertEqual(after["trainer.test_freq"], -1)
                self.assertEqual(after["trainer.save_freq"], -1)
                self.assertFalse(after["trainer.val_before_train"])
                for key in ("data.train_files", "data.seed", "data.max_prompt_length", "data.max_response_length",
                            "actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.kl_loss_coef",
                            "algorithm.adv_estimator", "critic.enable"):
                    self.assertEqual(before[key], after[key])
        self.assertEqual(base.to_dict(), original)
        self.assertEqual(base.steps, 100)

    def test_invalid_performance_probes_are_rejected(self):
        from search_r1_framework.performance import PerformanceProbe
        for kwargs in ({"micro_batch": 3}, {"micro_batch": 64}, {"measured_steps": 0},
                       {"warmup_steps": 0}, {"vllm_memory_fraction": 1.0},
                       {"cpu_offload": "False"}, {"max_num_seqs": 64}):
            with self.assertRaises(ValueError):
                PerformanceProbe(**kwargs).settings_for(Settings())

    def test_warmup_exclusion_and_gpu_measurement_windows(self):
        from search_r1_framework.performance import PerformanceProbe, summarize_performance
        probe = PerformanceProbe(warmup_steps=1, measured_steps=2)
        rows = [
            {"step": 1, "wall_time": 100, "metrics": {"timing_s/step": 90}},
            {"step": 2, "wall_time": 110, "metrics": {"timing_s/step": 10}},
            {"step": 3, "wall_time": 130, "metrics": {"timing_s/step": 20}},
        ]
        samples = [{"time": t, "gpu_name": "Test GPU", "memory_used_mib": mem,
                    "memory_total_mib": 100, "gpu_utilization_pct": 50, "power_w": 20}
                   for t, mem in ((50, 95), (105, 40), (125, 60))]
        report = summarize_performance(rows, samples, probe, 32)
        self.assertTrue(report["complete_measurement"])
        self.assertEqual(report["mean_step_seconds"], 15)
        self.assertAlmostEqual(report["records_per_second"], 64 / 30)
        self.assertEqual(report["measured_window_gpu"]["samples"], 2)
        self.assertEqual(report["measured_window_gpu"]["sampled_peak_memory_mib"], 60)
        self.assertEqual(report["measured_window_gpu"]["sampled_min_free_memory_mib"], 40)

    def test_partial_or_invalid_metrics_cannot_pass(self):
        from search_r1_framework.performance import PerformanceProbe, summarize_performance
        probe = PerformanceProbe(warmup_steps=1, measured_steps=2)
        for rows in ([], [{"step": 2, "metrics": {"timing_s/step": 1}}],
                     [{"step": 2, "metrics": {"timing_s/step": 1}},
                      {"step": 3, "metrics": {"timing_s/step": float("nan")}}]):
            self.assertFalse(summarize_performance(rows, [], probe, 32)["complete_measurement"])

    def test_gpu_missing_values_are_not_zero(self):
        from search_r1_framework.performance import parse_gpu_sample, gpu_summary
        sample = parse_gpu_sample("NVIDIA H100, 81920, 2048, 25, [N/A], 700\n", 100)
        self.assertEqual(sample["memory_used_mib"], 2048)
        self.assertIsNone(sample["power_w"])
        self.assertIsNone(gpu_summary([sample])["mean_power_w"])
        self.assertIsNone(gpu_summary([])["mean_gpu_utilization_pct"])

    def test_oom_and_other_errors_are_distinguished(self):
        from search_r1_framework.performance import PerformanceProbe, write_report
        with tempfile.TemporaryDirectory() as directory:
            sampler = SimpleNamespace(samples=[], errors=[])
            with patch("search_r1_framework.performance.read_native_metrics", return_value=[]):
                report = write_report(directory, PerformanceProbe(), Settings(), "ppo", sampler, 10,
                                      RuntimeError("CUDA out of memory"))
                self.assertEqual(report["failure_kind"], "gpu_out_of_memory")
                report = write_report(directory, PerformanceProbe(), Settings(), "ppo", sampler, 10,
                                      RuntimeError("Connection failed"))
                self.assertEqual(report["failure_kind"], "other_error")
                self.assertEqual(report["status"], "failed")
                self.assertFalse((Path(directory) / "finished.json").exists())

    def test_cloud_probe_routes_to_same_trainer_without_formal_checkpoint(self):
        from contextlib import ExitStack
        from unittest.mock import MagicMock
        import importlib.util
        from search_r1_framework.performance import PerformanceProbe
        script = Path(__file__).resolve().parents[1] / "search_r1_framework_modal.py"
        modal = ModuleType("modal")
        modal.Image = MagicMock()
        modal.Volume = MagicMock()
        def decorate(*args, **kwargs):
            def wrap(function):
                function.local = function
                return function
            return wrap
        modal.App = lambda *args: SimpleNamespace(function=decorate, cls=decorate)
        modal.enter = modal.method = decorate
        with patch.dict(sys.modules, {"modal": modal}):
            spec = importlib.util.spec_from_file_location("performance_modal_test", script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            directory = Path(temporary)
            def remap(value):
                value = str(value)
                return directory / value.lstrip("/") if value.startswith(("/data", "/results")) else Path(value)
            module.Path = remap
            module.start_search_bridge = lambda *args, **kwargs: (None, None, [])
            sampler = SimpleNamespace(start=lambda: None, stop=lambda: None, samples=[], errors=[])
            stack.enter_context(patch("search_r1_framework.performance.GpuSampler", return_value=sampler))
            stack.enter_context(patch("search_r1_framework.performance.read_native_metrics", return_value=[
                {"step": step, "metrics": {"timing_s/step": 10.0}} for step in range(1, 11)]))
            stack.enter_context(patch("search_r1_framework.data.validate_prepared", return_value={
                "model_path": "/model", "model_revision": "revision", "files_sha256": {"train.parquet": "hash"}}))
            stack.enter_context(patch("importlib.metadata.version", return_value="test"))
            stack.enter_context(patch.dict("os.environ", {}))
            trainer = stack.enter_context(patch("search_r1_framework.runner.run_framework"))
            final_actor_mock = stack.enter_context(patch("search_r1_framework.runner.final_actor"))
            result = module.run_performance_test("ppo", "perf-test", "prep", Settings().to_dict(),
                                                 PerformanceProbe(micro_batch=4).to_dict())
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("deadline_unix", trainer.call_args.kwargs)
            actual = parsed(trainer.call_args.args[0])
            self.assertEqual(actual["algorithm.adv_estimator"], "gae")
            self.assertEqual(actual["trainer.save_freq"], -1)
            final_actor_mock.assert_not_called()
            self.assertTrue((directory / "results/perf-test/performance.json").exists())
            self.assertFalse((directory / "results/perf-test/finished.json").exists())
            def recovered(*args, **kwargs):
                (Path(args[1]) / "search_requests.jsonl").write_text(
                    json.dumps({"kind": "retriever_restarted"}) + "\n")
            trainer.side_effect = recovered
            recovered_report = module.run_performance_test(
                "grpo", "perf-recovered", "prep", Settings().to_dict(), {})
            self.assertEqual(recovered_report["status"], "completed_with_warnings")
            self.assertFalse(recovered_report["comparison_eligible"])
            with self.assertRaises(FileExistsError):
                module.run_performance_test("ppo", "perf-test", "prep", Settings().to_dict(), {})
            trainer.side_effect = RuntimeError("Actual worker failure")
            with self.assertRaises(RuntimeError):
                module.run_performance_test("ppo", "perf-budget", "prep", Settings().to_dict(), {})
            failed = json.loads((directory / "results/perf-budget/performance.json").read_text())
            self.assertEqual(failed["failure_kind"], "other_error")
            self.assertFalse(failed["comparison_eligible"])
            self.assertIn("retriever_startup", failed["phase_seconds"])
            self.assertEqual(failed["run_name"], "perf-budget")


if __name__ == "__main__":
    unittest.main()
