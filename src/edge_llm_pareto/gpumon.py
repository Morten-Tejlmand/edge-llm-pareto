"""Sample GPU activity, clocks and power; tolerate unsupported GB10 counters.

NVML activity percentages are not measurements of compute or bandwidth saturation.
On GB10, system memory use is recorded separately from GPU framebuffer memory.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class GpuSamples:
    sm_util: list[int] = field(default_factory=list)
    mem_util: list[int] = field(default_factory=list)
    mem_used_mib: list[int] = field(default_factory=list)
    system_used_mib: list[int] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)
    # Sampled because a throttled clock makes every timing incomparable, and
    # utilization does not reveal it -- a GPU at 7% of peak clock still reports
    # ~100% SM utilization.
    sm_clock_mhz: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        def stats(xs: list, name: str) -> dict:
            if not xs:
                return {}
            return {
                f"{name}_mean": round(sum(xs) / len(xs), 1),
                f"{name}_max": round(max(xs), 1),
            }

        out: dict = {"samples": len(self.sm_util)}
        out.update(stats(self.sm_util, "sm_util_pct"))
        out.update(stats(self.mem_util, "mem_util_pct"))
        out.update(stats(self.mem_used_mib, "mem_used_mib"))
        out.update(stats(self.system_used_mib, "system_used_mib"))
        out.update(stats(self.power_w, "power_w"))
        out.update(stats(self.sm_clock_mhz, "sm_clock_mhz"))
        return out


class GpuMonitor:
    """Samples NVML on a thread for the duration of a benchmark run."""

    def __init__(self, index: int = 0, interval_s: float = 0.1):
        self.index = index
        self.interval_s = interval_s
        self.samples = GpuSamples()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._err: str | None = None

    def __enter__(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                name = pynvml.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                unified = "GB10" in name.upper()
                while not self._stop.is_set():
                    # An unavailable framebuffer counter must not discard the
                    # clock, power or utilization measurements from this sample.
                    try:
                        u = pynvml.nvmlDeviceGetUtilizationRates(h)
                        self.samples.sm_util.append(u.gpu)
                        self.samples.mem_util.append(u.memory)
                    except pynvml.NVMLError as exc:
                        self._err = str(exc)
                    try:
                        if unified:
                            from .hardware import system_memory_mib
                            total, available = system_memory_mib()
                            self.samples.system_used_mib.append(total - available)
                        else:
                            m = pynvml.nvmlDeviceGetMemoryInfo(h)
                            self.samples.mem_used_mib.append(m.used // 1024**2)
                    except (pynvml.NVMLError, OSError, ValueError, KeyError) as exc:
                        self._err = str(exc)
                    try:
                        self.samples.power_w.append(
                            pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                    except pynvml.NVMLError:
                        pass
                    try:
                        self.samples.sm_clock_mhz.append(
                            pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                    except pynvml.NVMLError:
                        pass
                    self._stop.wait(self.interval_s)
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:  # nvml missing or no permission -- bench still runs
            self._err = f"{type(e).__name__}: {e}"

    @property
    def error(self) -> str | None:
        return self._err
