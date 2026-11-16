"""Backend orchestration tests; no model downloads or GPU execution required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto import bench, hardware, serve

ROOT = Path(__file__).resolve().parents[1]


class GgufTests(unittest.TestCase):
    def hub(self, names):
        info = types.SimpleNamespace(sha="immutable-commit", siblings=[
            types.SimpleNamespace(rfilename=n) for n in names])
        return types.SimpleNamespace(
            HfApi=Mock(return_value=Mock(model_info=Mock(return_value=info))),
            hf_hub_download=Mock(side_effect=lambda **kw: "/cache/" + kw["filename"]))

    def test_download_resolves_nested_filename_and_pins_revision(self):
        hub = self.hub(["Q4/model-Q4.gguf", "Q8/model-Q8.gguf"])
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            result = serve.download_gguf("owner/model", "model-Q4.gguf", "release")
        self.assertEqual(result["path"], "/cache/Q4/model-Q4.gguf")
        self.assertEqual(result["revision"], "immutable-commit")
        hub.HfApi().model_info.assert_called_once_with("owner/model", revision="release")
        hub.hf_hub_download.assert_called_once_with(
            repo_id="owner/model", filename="Q4/model-Q4.gguf", revision="immutable-commit")

    def test_download_fetches_all_shards_and_returns_first(self):
        files = [f"BF16/model-BF16-{i:05d}-of-00002.gguf" for i in (1, 2)]
        hub = self.hub(files)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            result = serve.download_gguf("owner/model", "model-BF16.gguf")
        self.assertEqual(result["path"], "/cache/" + files[0])
        self.assertEqual(result["files"], files)
        self.assertEqual(hub.hf_hub_download.call_count, 2)

    def test_missing_ambiguous_or_incomplete_files_fail_before_download(self):
        for files in ([], ["a/model.gguf", "b/model.gguf"],
                      ["model-00001-of-00002.gguf"]):
            with self.subTest(files=files):
                hub = self.hub(files)
                with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaises(ValueError):
                    serve.download_gguf("owner/model", "model.gguf")
                hub.hf_hub_download.assert_not_called()


class LlamaLaunchTests(unittest.TestCase):
    def test_context_alias_cuda_and_cache_controls(self):
        cmd = serve.llama_server_command("/models/model.gguf", parallel=4,
                                         ctx_size=4096, alias="test-model")
        self.assertEqual(cmd[cmd.index("-c") + 1], "16384")
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "4")
        self.assertEqual(cmd[cmd.index("--alias") + 1], "test-model")
        self.assertEqual(cmd[cmd.index("--device") + 1], "CUDA0")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertIn("--no-cache-prompt", cmd)
        self.assertIn("--no-kv-unified", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "none")
        self.assertNotIn("--gpu-memory-utilization", cmd)



class LlamaSweepTests(unittest.TestCase):
    def run_cli(self, directory, *, download_error=False, ready=True, workloads=False, extra=None):
        spec = importlib.util.spec_from_file_location("llama_sweep", ROOT / "scripts/sweep.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        gpu = hardware.Gpu("NVIDIA GB10", 128 * 1024, 100 * 1024, "12.1", True)
        shape = hardware.ModelShape("test", 28, 2048, 16, 8, 128, 151936, 6144, True)
        ps = hardware.PowerState(None, None, None, None, None, [])
        handle = Mock(base_url="http://localhost/v1")
        handle.__enter__ = Mock(return_value=handle)
        handle.__exit__ = Mock(return_value=None)
        # Force two slots while sending four concurrent clients, testing queueing.
        capacity = hardware.Budget("test", 100, 4096, 1000, 8192, True)
        gguf = dict(path="/cache/test.gguf", repo="test", revision="abc", files=["test.gguf"])
        row = dict(output_tok_s=10, per_agent_tok_s=10, ttft_s_mean=.1, failed=0,
                   concurrency=4)
        with patch.object(cli, "probe_gpu", return_value=gpu), \
                patch.object(cli, "probe_power_state", return_value=ps), \
                patch.object(cli.ModelShape, "from_hf", return_value=shape), \
                patch.object(cli, "budget", return_value=capacity), \
                patch.object(serve, "resolve_executable", return_value="/bin/llama-server"), \
                patch.object(serve, "executable_version", return_value="build-test"), \
                patch.object(serve, "download_gguf", return_value=gguf,
                             side_effect=ValueError("missing file") if download_error else None) as download, \
                patch.object(serve, "start", return_value=handle) as start, \
                patch.object(serve, "wait_ready", return_value=ready), \
                patch.object(serve, "tail_log", return_value="unsupported kernel"), \
                patch.object(bench, "run_concurrency_level", AsyncMock(return_value=row)) as run, \
                patch.object(sys, "argv", ["sweep.py", "--model",
                                          "qwen3-4b-2507", "--concurrency", "4", "--results-dir", directory] + (["--workloads", str(ROOT / "configs/workloads/pilot.yaml")] if workloads else []) + (extra or [])):
            rc = cli.main()
        saved = json.loads(next(Path(directory).glob("*.jsonl")).read_text().splitlines()[0])
        return rc, saved, start, run, download, handle

    def test_default_gguf_download_launch_request_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            rc, saved, start, run, download, handle = self.run_cli(d)
        self.assertEqual(rc, 0)
        download.assert_called_once_with("unsloth/Qwen3-4B-Instruct-2507-GGUF",
                                         "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf", "main")
        cmd = start.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "2")
        self.assertEqual(cmd[cmd.index("-c") + 1], "8192")
        self.assertEqual(run.call_args.args[1], cmd[cmd.index("--alias") + 1])
        self.assertEqual(saved["engine"], "llama_cpp")
        self.assertEqual(saved["server_parallel"], 2)
        self.assertEqual(saved["gguf"]["revision"], "abc")
        self.assertEqual(saved["llama_cpp_version"], "build-test")
        self.assertIsNone(saved["gpu_memory_utilization"])
        handle.__exit__.assert_called_once()

    def test_workload_file_reaches_benchmark_as_dictionaries(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, _, run, _, _ = self.run_cli(d, workloads=True)
        self.assertEqual(rc, 0)
        self.assertGreater(run.await_count, 1)
        for call in run.await_args_list:
            self.assertIsInstance(call.args[3], dict)
            self.assertGreater(call.args[3]["prompt_tokens"], 0)

    def test_explicit_slots_and_fixed_prompts_reach_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'prompts.jsonl'
            path.write_text(json.dumps(dict(id='one', prompt='Continue this passage.')) + '\n')
            # Prompt input is not a result file.
            prompts = Path(d) / 'input'
            prompts.mkdir()
            path.rename(prompts / path.name)
            rc, saved, start, run, _, _ = self.run_cli(d, extra=[
                '--server-parallel', '1', '--prompt-dataset', str(prompts / path.name)])
            self.assertEqual(rc, 0)
            cmd = start.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--parallel') + 1], '1')
            self.assertEqual(saved['server_parallel'], 1)
            self.assertEqual(run.call_args.args[3]['prompts'][0]['prompt'], 'Continue this passage.')

    def test_explicit_slots_above_capacity_fail(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, 'exceed estimated capacity'):
            self.run_cli(d, extra=['--server-parallel', '3'])

    def test_download_failure_is_saved_without_launch(self):
        with tempfile.TemporaryDirectory() as d:
            rc, saved, start, run, _, _ = self.run_cli(d, download_error=True)
        self.assertEqual(rc, 1)
        self.assertTrue(saved["skipped"])
        self.assertIn("gguf_download_failed", saved["reason"])
        start.assert_not_called()
        run.assert_not_awaited()

    def test_server_failure_is_saved_and_process_cleaned_up(self):
        with tempfile.TemporaryDirectory() as d:
            rc, saved, _, run, _, handle = self.run_cli(d, ready=False)
        self.assertEqual(rc, 1)
        self.assertEqual(saved["reason"], "server_not_ready")
        run.assert_not_awaited()
        handle.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()

class DefaultRuntimeCheckTests(unittest.TestCase):
    def test_check_uses_llama_server_without_python_inference_libraries(self):
        spec = importlib.util.spec_from_file_location('check_cli', ROOT / 'scripts/check_spark.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        gpu = hardware.Gpu('NVIDIA GB10', 128000, 0, '12.1', True)
        with patch.object(cli, 'probe_gpu', return_value=gpu), \
                patch.object(cli.serve, 'resolve_executable', return_value='/bin/llama-server'), \
                patch.object(cli.serve, 'executable_version', return_value='test-build'), \
                patch.object(cli.subprocess, 'run', return_value=Mock(stdout='CUDA0: NVIDIA GB10', stderr='')) as run, \
                patch.dict(sys.modules, {'torch': None}), \
                patch.object(sys, 'argv', ['check_spark.py']):
            cli.main()
        self.assertEqual(run.call_args.args[0], ['/bin/llama-server', '--list-devices'])

    def test_cpu_only_server_is_rejected(self):
        spec = importlib.util.spec_from_file_location('check_cli', ROOT / 'scripts/check_spark.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        gpu = hardware.Gpu('NVIDIA GB10', 128000, 0, '12.1', True)
        with patch.object(cli, 'probe_gpu', return_value=gpu), \
                patch.object(cli.serve, 'resolve_executable', return_value='/bin/llama-server'), \
                patch.object(cli.serve, 'executable_version', return_value='test-build'), \
                patch.object(cli.subprocess, 'run', return_value=Mock(stdout='CPU', stderr='')), \
                patch.object(sys, 'argv', ['check_spark.py']):
            with self.assertRaisesRegex(SystemExit, 'does not list CUDA0'):
                cli.main()
