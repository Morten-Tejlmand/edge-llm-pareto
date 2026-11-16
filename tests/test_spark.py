import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from edge_llm_pareto import bench, telemetry, hardware, serve

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report", ROOT / "scripts/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class SparkMemoryTests(unittest.TestCase):
    def setUp(self):
        self.shape = hardware.ModelShape("test", 28, 2048, 16, 8, 128,
                                         151936, 6144, True)

    def test_system_memory_uses_available_not_free_or_swap(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "meminfo"
            p.write_text("MemTotal: 134217728 kB\nMemFree: 1024 kB\n"
                         "MemAvailable: 104857600 kB\nSwapFree: 999999 kB\n")
            self.assertEqual(hardware.system_memory_mib(str(p)), (131072, 102400))

    def test_gb10_smi_never_queries_framebuffer_memory(self):
        with patch("subprocess.run", return_value=types.SimpleNamespace(
                stdout="NVIDIA GB10, 12.1\n")) as run, patch.object(
                    hardware, "system_memory_mib", return_value=(128000, 100000)):
            gpu = hardware._probe_smi()
        self.assertTrue(gpu.unified_memory)
        self.assertEqual(gpu.free_mib, 100000)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--query-gpu=name,compute_cap", run.call_args.args[0])

    def test_discrete_gpu_still_uses_device_memory(self):
        outputs = [types.SimpleNamespace(stdout="RTX 4050, 8.9"),
                   types.SimpleNamespace(stdout="6141, 400")]
        with patch("subprocess.run", side_effect=outputs):
            gpu = hardware._probe_smi()
        self.assertFalse(gpu.unified_memory)
        self.assertEqual(gpu.free_mib, 5741)

    def test_busy_host_limits_budget(self):
        gpu = hardware.Gpu("GB10", 128 * 1024, 100 * 1024, "12.1", True)
        b = hardware.budget(self.shape, gpu, "bf16", 16)
        # 28 GiB available minus 16 GiB headroom leaves a 12 GiB process cap.
        self.assertEqual(b.kv_mib, 12 * 1024 - b.weights_mib - 4096)

    def test_idle_host_uses_utilization_cap(self):
        gpu = hardware.gb10_preview()
        b = hardware.budget(self.shape, gpu, "bf16", 16)
        self.assertEqual(b.kv_mib, int(gpu.total_mib * .7) - b.weights_mib - 4096)
        self.assertGreater(b.concurrency_at(4096), 0)

    def test_no_memory_does_not_fit(self):
        gpu = hardware.Gpu("GB10", 128000, 120000, "12.1", True)
        b = hardware.budget(self.shape, gpu, "bf16", 16)
        self.assertFalse(b.fits)
        self.assertEqual(b.concurrency_at(4096), 0)

    def test_invalid_utilization_rejected(self):
        for value in (0, -1, 1, float("nan")):
            with self.assertRaises(ValueError):
                hardware.budget(self.shape, hardware.gb10_preview(), "bf16", 16,
                                gpu_memory_utilization=value)

    def test_launch_uses_context_slots_and_loopback(self):
        cmd = serve.llama_server_command("test.gguf", parallel=2, ctx_size=4096)
        self.assertEqual(cmd[cmd.index("-c") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")


class TelemetryTests(unittest.TestCase):
    def test_unsupported_utilization_preserves_clock_and_system_memory(self):
        class NvmlError(Exception):
            pass
        fake = types.SimpleNamespace(
            NVMLError=NvmlError, NVML_CLOCK_SM=0,
            nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(return_value=1),
            nvmlDeviceGetName=Mock(return_value="NVIDIA GB10"),
            nvmlDeviceGetUtilizationRates=Mock(side_effect=NvmlError("unsupported")),
            nvmlDeviceGetMemoryInfo=Mock(side_effect=AssertionError("framebuffer queried")),
            nvmlDeviceGetPowerUsage=Mock(return_value=100000),
            nvmlDeviceGetClockInfo=Mock(return_value=2000),
        )
        with patch.dict(sys.modules, {"pynvml": fake}), patch.object(
                telemetry, "is_jetson", return_value=False), patch.object(
                hardware, "system_memory_mib", return_value=(128000, 100000)):
            with telemetry.PowerMonitor(interval=60) as mon:
                pass
        self.assertEqual(mon.summary()["sm_clock_mhz_mean"], 2000)
        self.assertEqual(mon.summary()["system_used_mib_mean"], 28000)
        self.assertNotIn("mem_used_mib_mean", mon.summary())
        self.assertEqual(len(mon.samples), 2)
        fake.nvmlInit.assert_called_once()
        fake.nvmlShutdown.assert_called_once()


class ResultTests(unittest.TestCase):
    def test_metadata_is_saved_before_return(self):
        row = dict(output_tok_s=10, per_agent_tok_s=10, ttft_s_mean=.1, failed=0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "result.jsonl"
            with patch.object(bench, "run_concurrency_level", AsyncMock(return_value=row)):
                asyncio.run(bench.sweep("http://localhost/v1", "test", [1],
                                        bench.make_workload(), "bf16", str(out),
                                        metadata={"kv_ceiling_agents": 42, "throttled": True}))
            saved = json.loads(out.read_text())
        self.assertEqual(saved["kv_ceiling_agents"], 42)
        self.assertTrue(saved["throttled"])

    def test_failed_or_throttled_results_do_not_dominate_valid_result(self):
        valid = dict(output_tok_s=10, per_agent_tok_s=10)
        throttled = dict(output_tok_s=100, per_agent_tok_s=100, throttled=True)
        failed = dict(output_tok_s=200, per_agent_tok_s=200, failed=1)
        self.assertEqual(report.pareto([valid, throttled, failed]), [valid])


class SweepIntegrationTests(unittest.TestCase):
    def test_busy_gb10_budget_reaches_slots_and_saved_result(self):
        spec = importlib.util.spec_from_file_location("sweep_cli", ROOT / "scripts/sweep.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        gpu = hardware.Gpu("NVIDIA GB10", 128 * 1024, 100 * 1024, "12.1", True)
        shape = hardware.ModelShape("test", 28, 2048, 16, 8, 128, 151936, 6144, True)
        ps = hardware.PowerState(None, None, None, None, None, [])
        handle = Mock(base_url="http://localhost/v1")
        handle.__enter__ = Mock(return_value=handle)
        handle.__exit__ = Mock(return_value=None)
        row = dict(output_tok_s=10, per_agent_tok_s=10, ttft_s_mean=.1, failed=0,
                   concurrency=1)
        with tempfile.TemporaryDirectory() as d, patch.object(cli, "probe_gpu", return_value=gpu), \
                patch.object(cli, "probe_power_state", return_value=ps), \
                patch.object(cli.ModelShape, "from_hf", return_value=shape), \
                patch.object(cli.serve, "executable_version", return_value="test"), \
                patch.object(cli.serve, "download_gguf", return_value=dict(path="test.gguf", revision="test-sha", files=["test.gguf"])), \
                patch.object(cli.serve, "resolve_executable", return_value="/bin/llama-server"), \
                patch.object(cli.serve, "start", return_value=handle) as start, \
                patch.object(cli.serve, "wait_ready", return_value=True), \
                patch.object(bench, "run_concurrency_level", AsyncMock(return_value=row)) as run, \
                patch.object(sys, "argv", ["sweep.py", "--engine", "llama_cpp", "--results-dir", d]):
            self.assertEqual(cli.main(), 0)
            start.assert_called_once()
            run.assert_awaited_once()
            self.assertEqual(run.call_args.args[2], 1)
            self.assertEqual(run.call_args.args[4], 1)
            cmd = start.call_args.args[0]
            self.assertEqual(cmd[cmd.index("-m") + 1], "test.gguf")
            self.assertEqual(cmd[cmd.index("--parallel") + 1], "1")
            saved = json.loads(next(Path(d).glob("*.jsonl")).read_text())
            self.assertGreater(saved["kv_ceiling_agents"], 0)
            self.assertEqual(saved["hardware"]["unified_memory"], True)
            self.assertIsNone(saved["gpu_memory_utilization"])
            self.assertEqual(saved["memory_policy"], "estimated_slots")



class UnifiedMonitorTests(unittest.TestCase):
    def test_missing_power_preserves_other_counters_and_cleanup(self):
        fake = types.SimpleNamespace(
            NVML_CLOCK_SM=0, nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(return_value=1),
            nvmlDeviceGetName=Mock(return_value="RTX"),
            nvmlDeviceGetPowerUsage=Mock(side_effect=RuntimeError("unsupported")),
            nvmlDeviceGetClockInfo=Mock(return_value=2000),
            nvmlDeviceGetTemperature=Mock(return_value=45),
            nvmlDeviceGetCurrentClocksThrottleReasons=Mock(return_value=0),
            nvmlDeviceGetUtilizationRates=Mock(
                return_value=types.SimpleNamespace(gpu=80, memory=40)),
            nvmlDeviceGetMemoryInfo=Mock(
                return_value=types.SimpleNamespace(used=1024**3)),
        )
        with patch.dict(sys.modules, {"pynvml": fake}), patch.object(
                telemetry, "is_jetson", return_value=False):
            with self.assertRaisesRegex(ValueError, "workload failed"):
                with telemetry.PowerMonitor(interval=60) as mon:
                    raise ValueError("workload failed")
        self.assertFalse(mon.thread.is_alive())
        fake.nvmlShutdown.assert_called_once()
        self.assertEqual(mon.summary()["sm_util_pct_mean"], 80)
        self.assertEqual(mon.summary()["mem_used_mib_mean"], 1024)
        self.assertNotIn("power_w_mean", mon.summary())
        self.assertTrue(all(s["power_w"] is None for s in mon.samples))


if __name__ == "__main__":
    unittest.main()
