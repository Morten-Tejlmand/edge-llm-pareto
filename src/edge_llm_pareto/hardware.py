"""GPU detection and approximate memory budgets for GB10 unified memory."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

MIB = 1024**2
GIB = 1024**3

# Starting estimates for GB10, to be calibrated from server startup logs.
RUNTIME_OVERHEAD_MIB = 4096
DRIVER_RESERVED_MIB = 0
HOST_RESERVE_MIB = 16384
DEFAULT_GPU_MEMORY_UTILIZATION = 0.70


@dataclass(frozen=True)
class Gpu:
    name: str
    total_mib: int
    used_mib: int
    compute_cap: str
    unified_memory: bool = False
    memory_source: str = "device"

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


@dataclass(frozen=True)
class ModelShape:
    """The parts of a HF config.json that drive memory use."""

    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    intermediate: int
    tied_embeddings: bool

    @classmethod
    def from_config(cls, cfg: dict, name: str) -> "ModelShape":
        # Multimodal checkpoints keep the language architecture in text_config.
        # Hybrid attention is approximated as full attention at every layer;
        # recurrent-state/vision allocations still need runtime validation.
        cfg = cfg.get("text_config", cfg)
        heads = cfg["num_attention_heads"]
        return cls(
            name=name,
            layers=cfg["num_hidden_layers"],
            hidden=cfg["hidden_size"],
            heads=heads,
            kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            vocab=cfg["vocab_size"],
            intermediate=cfg["intermediate_size"],
            tied_embeddings=cfg.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_hf(cls, repo: str) -> "ModelShape":
        url = f"https://huggingface.co/{repo}/raw/main/config.json"
        with urllib.request.urlopen(url, timeout=30) as r:
            return cls.from_config(json.load(r), repo)

    # --- parameter accounting, split because quantizers treat these differently ---

    @property
    def embedding_params(self) -> int:
        """Embeddings (and the LM head when untied)."""
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2)

    @property
    def linear_params(self) -> int:
        """Transformer-block linears -- the part a quantizer actually compresses."""
        attn_width = self.heads * self.head_dim   # != hidden_size on some models
        attn = (
            self.hidden * attn_width                             # q_proj
            + 2 * self.hidden * self.kv_heads * self.head_dim    # k_proj, v_proj
            + attn_width * self.hidden                           # o_proj
        )
        mlp = 3 * self.hidden * self.intermediate                # gate, up, down
        return self.layers * (attn + mlp)

    @property
    def total_params(self) -> int:
        return self.embedding_params + self.linear_params

    def kv_bytes_per_token(self, kv_dtype_bytes: int = 2) -> int:
        """K and V, every layer. The number that sets the concurrency ceiling."""
        return self.layers * self.kv_heads * self.head_dim * 2 * kv_dtype_bytes


@dataclass(frozen=True)
class Budget:
    rung: str
    weights_mib: int
    overhead_mib: int
    kv_mib: int
    kv_tokens: int
    fits: bool

    def concurrency_at(self, context_tokens: int) -> int:
        """How many agents can hold `context_tokens` of KV simultaneously."""
        if not self.fits or context_tokens <= 0:
            return 0
        return self.kv_tokens // context_tokens


def weight_mib(shape: ModelShape, linear_bits: float, embedding_bits: float = 16.0) -> int:
    """Weight footprint. Quantizers compress block linears but usually leave
    embeddings near full precision -- which on small models is a large share of
    the total, so counting them separately matters at the low-bit end."""
    bits = shape.linear_params * linear_bits + shape.embedding_params * embedding_bits
    return int(bits / 8 / MIB)


def budget(
    shape: ModelShape,
    gpu: Gpu,
    rung: str,
    linear_bits: float,
    embedding_bits: float = 16.0,
    kv_dtype_bytes: int = 2,
    overhead_mib: int = RUNTIME_OVERHEAD_MIB,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    driver_reserved_mib: int = DRIVER_RESERVED_MIB,
    host_reserve_mib: int = HOST_RESERVE_MIB,
) -> Budget:
    """Estimate KV capacity within the configured process and available-memory caps.

    On GB10 the CPU and GPU share physical RAM. MemAvailable includes reclaimable
    cache; keep additional host headroom. Runtime overhead is an estimate, not a
    measured GB10 calibration. Server startup remains the final fit check.
    """
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu memory utilization must be between 0 and 1")
    if min(overhead_mib, driver_reserved_mib, host_reserve_mib) < 0:
        raise ValueError("memory reservations must be nonnegative")
    w = weight_mib(shape, linear_bits, embedding_bits)
    usable = min(
        int(gpu_memory_utilization * (gpu.total_mib - driver_reserved_mib)),
        max(0, gpu.free_mib - (host_reserve_mib if gpu.unified_memory else 0)),
    )
    kv = usable - w - overhead_mib
    per_token = shape.kv_bytes_per_token(kv_dtype_bytes)
    return Budget(
        rung=rung,
        weights_mib=w,
        overhead_mib=overhead_mib,
        kv_mib=max(kv, 0),
        kv_tokens=max(kv, 0) * MIB // per_token,
        fits=kv > 0,
    )


def system_memory_mib(path: str = "/proc/meminfo") -> tuple[int, int]:
    """Return physical total and available RAM, excluding swap."""
    from pathlib import Path

    fields = dict(line.split(":", 1) for line in Path(path).read_text().splitlines())
    total = int(fields["MemTotal"].split()[0]) // 1024
    available = int(fields["MemAvailable"].split()[0]) // 1024
    return total, available


def gb10_preview() -> Gpu:
    """Nominal 128 GiB planning profile; never used to launch a server."""
    return Gpu("NVIDIA GB10 (planning estimate)", 128 * 1024, 0, "12.1",
               True, "nominal; no live memory probe")


def _gpu_from_values(name: str, cap: str, total=None, used=None) -> Gpu:
    if "GB10" in name.upper():
        total_mib, available_mib = system_memory_mib()
        return Gpu(name, total_mib, total_mib - available_mib, cap, True,
                   "/proc/meminfo: MemTotal and MemAvailable")
    return Gpu(name, int(total), int(used), cap)


def probe_gpu(index: int = 0) -> Gpu:
    """Identify GB10 before requesting unsupported framebuffer memory counters."""
    from .telemetry import is_jetson
    if is_jetson():
        from pathlib import Path
        total, available = system_memory_mib()
        name = Path("/proc/device-tree/model").read_text().strip("\x00\n")
        return Gpu(name, total, total - available, "unknown", True, "/proc/meminfo")
    try:
        return _probe_nvml(index)
    except Exception:
        return _probe_smi(index)


def _probe_smi(index: int = 0) -> Gpu:
    import subprocess

    def query(fields):
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
             "-i", str(index)], capture_output=True, text=True, check=True,
            timeout=15,
        ).stdout.strip().split(",")

    name, cap = [x.strip() for x in query("name,compute_cap")]
    if "GB10" in name.upper():
        return _gpu_from_values(name, cap)
    total, used = query("memory.total,memory.used")
    return _gpu_from_values(name, cap, total, used)


def _probe_nvml(index: int = 0) -> Gpu:
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        cap = f"{major}.{minor}"
        if "GB10" in name.upper():
            return _gpu_from_values(name, cap)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return _gpu_from_values(name, cap, mem.total // MIB, mem.used // MIB)
    finally:
        pynvml.nvmlShutdown()


def add_memory_arguments(parser) -> None:
    parser.add_argument("--gpu-mem-util", type=float,
                        default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--host-reserve-gib", type=float, default=16,
                        help="extra available RAM to retain for the host on GB10")
    parser.add_argument("--runtime-overhead-gib", type=float, default=4,
                        help="estimated runtime memory beyond weights and KV cache")


def memory_options(args) -> dict:
    if not 0 < args.gpu_mem_util < 1:
        raise ValueError("--gpu-mem-util must be between 0 and 1")
    if args.host_reserve_gib < 0 or args.runtime_overhead_gib < 0:
        raise ValueError("memory reservations must be nonnegative")
    return dict(gpu_memory_utilization=args.gpu_mem_util,
                host_reserve_mib=int(args.host_reserve_gib * 1024),
                overhead_mib=int(args.runtime_overhead_gib * 1024))


# --- benchmark preconditions -------------------------------------------------
#
# Learned the hard way: this laptop on battery clocks the GPU to 210 MHz out of
# 3105 MHz (SW Power Cap active, pstate P8) while still reporting 99% SM
# utilization. Throughput came out ~30x low and, worse, *every* rung looked
# compute bound -- which would have inverted the central finding of this project.
# Utilization alone cannot detect this. Clock and power state can.


@dataclass(frozen=True)
class PowerState:
    on_ac: bool | None
    power_w: float | None
    power_limit_w: float | None
    sm_clock_mhz: int | None
    sm_clock_max_mhz: int | None
    throttle_reasons: list[str]

    @property
    def clock_fraction(self) -> float | None:
        if not self.sm_clock_mhz or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz


def probe_power_state(index: int = 0) -> PowerState:
    import glob
    import subprocess

    def smi(query: str) -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits", "-i", str(index)],
                capture_output=True, text=True, check=True, timeout=15,
            ).stdout.strip()
            return [x.strip() for x in out.split(",")]
        except Exception:
            return []

    def num(v: str, cast):
        try:
            return cast(v)
        except (ValueError, TypeError):
            return None

    vals = smi("power.draw,power.limit,clocks.sm,clocks.max.sm")
    power_w = num(vals[0], float) if len(vals) > 0 else None
    limit_w = num(vals[1], float) if len(vals) > 1 else None
    sm = num(vals[2], int) if len(vals) > 2 else None
    sm_max = num(vals[3], int) if len(vals) > 3 else None

    on_ac: bool | None = None
    for path in glob.glob("/sys/class/power_supply/A*/online"):
        try:
            with open(path) as f:
                on_ac = f.read().strip() == "1"
            break
        except OSError:
            pass

    reasons: list[str] = []
    try:
        import subprocess as sp

        q = sp.run(["nvidia-smi", "-q", "-d", "PERFORMANCE", "-i", str(index)],
                   capture_output=True, text=True, timeout=15).stdout
        in_block = False
        for line in q.splitlines():
            if "Clocks Event Reasons" in line:
                in_block = True
                continue
            if in_block:
                if "Counters" in line:
                    break
                if ":" in line:
                    name, _, val = line.partition(":")
                    if val.strip() == "Active" and "Idle" not in name:
                        reasons.append(name.strip())
    except Exception:
        pass

    return PowerState(on_ac, power_w, limit_w, sm, sm_max, reasons)


def check_benchmark_conditions(
    ps: PowerState, min_clock_fraction: float = 0.5, intentional_power_limit: bool = False,
    idle_power_w: float = 15,
) -> list[str]:
    """Reasons the machine is unfit to produce comparable timings."""
    problems = []
    if ps.on_ac is False:
        problems.append(
            "running on battery -- the GPU is power-capped and timings are meaningless"
        )
    throttles = ps.throttle_reasons
    # A power cap only limits a GPU that is drawing power. The GB10 can report
    # "SW Power Cap" while idle (~5 W, idle clocks), which is not throttling;
    # ignore it there. Thermal/hardware slowdowns always count, and per-sample
    # throttle reasons and clocks are still recorded during every trial.
    idle = ps.power_w is not None and ps.power_w < idle_power_w
    if any("Slowdown" in r or ("Power Cap" in r and not intentional_power_limit and not idle)
           for r in throttles):
        problems.append(f"GPU throttled: {', '.join(throttles)}")
    frac = ps.clock_fraction
    # Only meaningful under load; at idle a low clock is correct. The sweep
    # records sampled clocks during the run for later inspection.
    if not intentional_power_limit and frac is not None and frac < min_clock_fraction and (ps.power_w or 0) > idle_power_w:
        problems.append(
            f"SM clock at {ps.sm_clock_mhz}/{ps.sm_clock_max_mhz} MHz "
            f"({frac:.0%} of peak) under load"
        )
    return problems
