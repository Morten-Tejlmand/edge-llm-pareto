"""Launch, health-check and stop inference servers for the GB10 benchmark."""

from __future__ import annotations

import json
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
        # Signal the server process group.
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


def llama_server_command(
    gguf_path: str,
    *,
    n_gpu_layers: int = 999,
    ctx_size: int = 4096,
    parallel: int = 32,
    port: int = 8000,
    executable: str = "llama-server",
    alias: str = "benchmark",
    extra: list[str] | None = None,
) -> list[str]:
    """Build a llama.cpp command with ctx_size tokens per parallel slot."""
    cmd = [
        executable, "-m", gguf_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", alias,
        "--predict", "-1",
        "--device", "CUDA0",
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size * parallel),
        "--parallel", str(parallel),
        "--load-mode", "none",
        "--fit", "off",
        "--no-kv-unified",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--cache-ram", "0",
        "--no-cache-prompt",
        "--no-context-shift",
        "--jinja",
        # Keep Qwen thinking tokens in content so the stream timer observes them.
        "--reasoning-format", "none",
    ]
    if extra:
        cmd += extra
    return cmd


def download_gguf(repo: str, filename: str, revision: str = "main") -> dict:
    """Resolve a GGUF (including split files) at one immutable HF revision."""
    import re
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, revision=revision)
    names = [s.rfilename for s in info.siblings]
    if filename in names:
        selected = filename
    else:
        # Publishers sometimes move the same file into a quantization folder.
        matches = [n for n in names if pathlib.PurePosixPath(n).name == filename]
        if not matches:
            stem = re.escape(filename.removesuffix(".gguf"))
            matches = [n for n in names if re.fullmatch(
                rf"{stem}-00001-of-\d{{5}}\.gguf", pathlib.PurePosixPath(n).name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one GGUF matching {filename} in {repo}; found {matches}")
        selected = matches[0]

    split = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", selected)
    files = [selected]
    if split:
        prefix, _, count = split.groups()
        files = [f"{prefix}-{i:05d}-of-{count}.gguf" for i in range(1, int(count) + 1)]
        if not files or any(f not in names for f in files):
            raise ValueError(f"Incomplete GGUF shard set: {selected}")
    paths = [hf_hub_download(repo_id=repo, filename=f, revision=info.sha) for f in files]
    return {"path": paths[0], "repo": repo, "revision": info.sha, "files": files}


def resolve_hf_revision(repo: str, revision: str = "main") -> str:
    """Pin native weights before vLLM downloads them."""
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, revision=revision).sha


def vllm_server_command(model: str, *, revision: str, alias: str,
                        executable: str = "vllm", ctx_size: int = 4096,
                        parallel: int = 16, port: int = 8000,
                        gpu_memory_utilization: float = .7,
                        kv_cache_memory_bytes: int | None = None,
                        dtype: str = "bfloat16", quantization: str | None = None,
                        tokenizer: str | None = None,
                        tokenizer_revision: str | None = None) -> list[str]:
    """Native vLLM serving; GGUF requires the separate vLLM GGUF plugin.

    Leave compilation and continuous batching enabled. Explicit KV bytes honor
    the host-aware planner on unified memory; utilization alone uses GPU counters.
    """
    cmd = [executable, "serve", model, "--host", "127.0.0.1",
           "--port", str(port), "--served-model-name", alias,
           "--revision", revision, "--max-model-len", str(ctx_size),
           "--max-num-seqs", str(parallel), "--dtype", dtype,
           "--kv-cache-dtype", dtype,
           "--gpu-memory-utilization", str(gpu_memory_utilization),
           "--disable-prefix-caching", "--generation-config", "vllm"]
    if kv_cache_memory_bytes is not None:
        cmd += ["--kv-cache-memory-bytes", str(kv_cache_memory_bytes)]
    if quantization:
        cmd += ["--quantization", quantization]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
        if not os.path.isdir(tokenizer):  # a local directory has no Hub revision
            cmd += ["--tokenizer-revision", tokenizer_revision or revision]
    return cmd


def prepare_tokenizer_override(repo: str, revision: str, tokenizer_class: str,
                               root: str | os.PathLike | None = None) -> str:
    """Local copy of a repo's tokenizer with only `tokenizer_class` replaced.

    DeepSeek-R1-0528-Qwen3-8B declares `LlamaTokenizerFast`, but its vocabulary
    is Qwen3 byte-level BPE. transformers 5.x builds that class with a
    SentencePiece-style pre-tokenizer/decoder that overrides tokenizer.json, so
    vLLM drops spaces/newlines when encoding prompts and returns raw 'Ġ'/'Ċ'
    symbols when decoding. `PreTrainedTokenizerFast` loads tokenizer.json as
    published. Vocabulary, merges, special tokens and chat template are the
    repo's own files at the pinned revision; only the loader class changes.
    """
    from huggingface_hub import hf_hub_download

    root = pathlib.Path(root or pathlib.Path.home() / ".cache" / "edge-llm-pareto" / "tokenizers")
    out = root / f"{repo.replace('/', '--')}-{revision[:12]}-{tokenizer_class}"
    out.mkdir(parents=True, exist_ok=True)
    # config.json too: vLLM also parses the model config from the tokenizer path.
    for name in ("tokenizer.json", "config.json"):
        shutil.copyfile(hf_hub_download(repo, name, revision=revision), out / name)
    config = json.loads(pathlib.Path(
        hf_hub_download(repo, "tokenizer_config.json", revision=revision)).read_text())
    config["tokenizer_class"] = tokenizer_class
    (out / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    return str(out)


def executable_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], capture_output=True,
                            text=True, check=True, timeout=30)
    return (result.stdout + result.stderr).strip()


def resolve_executable(name: str) -> str:
    """Find `name` on PATH, else next to the running interpreter.

    Running as `.venv/bin/python scripts/sweep.py` does not put the venv's bin
    directory on PATH; also check beside the interpreter.
    """
    found = shutil.which(name)
    if found:
        return found
    sibling = pathlib.Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError(
        f"{name!r} not found on PATH or in {sibling.parent}. "
        "For llama-server build instructions, see docs/engines.md"
    )


def start(cmd: list[str], log_path: str, port: int = 8000,
          env: dict[str, str] | None = None) -> ServerHandle:
    cmd = [resolve_executable(cmd[0]), *cmd[1:]]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
    return ServerHandle(proc, f"http://127.0.0.1:{port}/v1", log_path)


def wait_ready(h: ServerHandle, timeout_s: float = 600.0, poll_s: float = 2.0) -> bool:
    """Allow time for model loading; return False if the server process dies."""
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
