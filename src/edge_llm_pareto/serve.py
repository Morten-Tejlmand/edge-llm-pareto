"""Launch, health-check and stop inference servers for the GB10 benchmark."""

from __future__ import annotations

import os
import shutil
import signal
import sys
import subprocess
import time
import pathlib
import urllib.error
import urllib.request


class ServerHandle:
    def __init__(self, proc: subprocess.Popen, base_url: str, log_path: str):
        self.proc = proc
        self.base_url = base_url
        self.log_path = log_path

    def stop(self, timeout: float = 30.0) -> None:
        if self.proc.poll() is not None:
            return
        # vLLM spawns worker processes; signal the whole group.
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait(timeout=10)

    def __enter__(self) -> "ServerHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def vllm_command(
    repo: str,
    *,
    quantization: str | None = None,
    max_model_len: int = 4096,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.70,
    kv_cache_dtype: str = "auto",
    kv_offloading_gib: float | None = None,
    enforce_eager: bool = False,
    port: int = 8000,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "vllm", "serve", repo,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--kv-cache-dtype", kv_cache_dtype,
        # Each benchmark request gets a unique prefix, but disable the cache
        # explicitly so a rerun cannot be contaminated by a warm cache.
        "--no-enable-prefix-caching",
    ]
    if kv_offloading_gib:
        # vLLM 0.27 replaced the old --swap-space with explicit KV offloading.
        # Spilling KV to host RAM raises the agent ceiling past what 6 GiB of
        # VRAM allows -- at the cost of PCIe transfers, so it is its own axis
        # rather than a default.
        cmd += ["--kv-offloading-size", str(kv_offloading_gib),
                "--kv-offloading-backend", "native"]
    if quantization:
        cmd += ["--quantization", quantization]
    if enforce_eager:
        # Saves ~400-600 MiB of CUDA graph pools but slows decode measurably.
        # Only for rungs that otherwise refuse to fit; note it in results.
        cmd += ["--enforce-eager"]
    if extra:
        cmd += extra
    return cmd


def llama_server_command(
    gguf_path: str,
    *,
    n_gpu_layers: int = 999,
    ctx_size: int = 4096,
    parallel: int = 32,
    port: int = 8000,
    extra: list[str] | None = None,
) -> list[str]:
    """llama.cpp equivalent. --parallel splits the KV cache into that many slots,
    so ctx_size here is *per slot* -- the analogue of vLLM's max_model_len."""
    cmd = [
        "llama-server", "-m", gguf_path,
        "--port", str(port),
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size * parallel),
        "--parallel", str(parallel),
        "--no-mmap",
    ]
    if extra:
        cmd += extra
    return cmd


def resolve_executable(name: str) -> str:
    """Find `name` on PATH, else next to the running interpreter.

    Running as `.venv/bin/python scripts/sweep.py` does not put the venv's bin
    directory on PATH, so a plain PATH lookup misses the `vllm` sitting right
    beside the interpreter that imported this module.
    """
    found = shutil.which(name)
    if found:
        return found
    sibling = pathlib.Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError(
        f"{name!r} not found on PATH or in {sibling.parent}. "
        f"For vllm: `uv sync`; for llama-server: see docs/engines.md"
    )


def find_cuda_home() -> str | None:
    """Locate a CUDA toolkit inside the venv.

    vLLM's warm-up JIT-compiles FlashInfer kernels, which needs `nvcc`. There is
    no system CUDA toolkit here, but the nvidia pip wheels torch pulls in ship a
    complete one (nvcc + headers + libs) under site-packages/nvidia/cuXX.
    FlashInfer only checks $CUDA_HOME and /usr/local/cuda, so it has to be
    pointed at that tree explicitly or startup dies after several minutes of
    otherwise-successful loading.
    """
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    site = pathlib.Path(sys.executable).parent.parent / "lib"
    for nvidia_dir in sorted(site.glob("python*/site-packages/nvidia")):
        # Prefer the highest CUDA major version present.
        for cu in sorted(nvidia_dir.glob("cu*"), reverse=True):
            if (cu / "bin" / "nvcc").exists():
                return str(cu)
    return None


def host_compiler_too_new() -> bool:
    """CUDA 13.3 rejects GCC > 15; Fedora 44 ships GCC 16.

    nvcc refuses to compile at all against a newer host compiler, so any JIT
    path is dead until either an older gcc is installed or the JIT is avoided.
    """
    try:
        out = subprocess.run(["gcc", "-dumpversion"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return int(out.split(".")[0]) > 15
    except Exception:
        return False


def server_env() -> dict[str, str]:
    env = dict(os.environ)
    cuda_home = find_cuda_home()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")

    # FlashInfer JIT-compiles its sampling kernels at startup. With GCC 16 as
    # the host compiler that build fails and takes the whole engine down, so
    # fall back to vLLM's native sampler. At temperature=0 sampling is argmax,
    # so this is not a meaningful throughput difference -- but it is recorded in
    # run metadata because it is a deviation from the default path.
    if host_compiler_too_new():
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def start(cmd: list[str], log_path: str, port: int = 8000) -> ServerHandle:
    cmd = [resolve_executable(cmd[0]), *cmd[1:]]
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=server_env(),
    )
    return ServerHandle(proc, f"http://127.0.0.1:{port}/v1", log_path)


def wait_ready(h: ServerHandle, timeout_s: float = 600.0, poll_s: float = 2.0) -> bool:
    """vLLM startup is slow (weight load + torch.compile + graph capture), so the
    default timeout is generous. Returns False if the process died instead."""
    health = h.base_url.rsplit("/v1", 1)[0] + "/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if h.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(poll_s)
    return False


def tail_log(h: ServerHandle, n: int = 40) -> str:
    try:
        with open(h.log_path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"
