# GB10 engine setup

## Native Python runtime

`pyproject.toml` and `uv.lock` define the Python 3.12 / vLLM 0.27.1
native environment. Run these commands from the repository root:

```bash
uv sync
.venv/bin/python scripts/check_spark.py
.venv/bin/python scripts/budget.py
.venv/bin/python scripts/sweep.py --model qwen3-1.7b --rung bf16 --concurrency 1,2
```

Native GB10 operation requires validating the installed driver, CUDA, PyTorch
and vLLM kernels on the Spark. The locked environment has not been validated
on GB10; successful dependency installation alone does not establish support.

`check_spark.py` checks GB10 detection and performs a small CUDA matrix operation.
A successful check is not proof that AWQ/GPTQ/FP8/bitsandbytes kernels work: test
the configured rungs individually before relying on a full sweep. Install the
optional bitsandbytes plugin with `uv sync --extra bnb` when needed. Unsupported
rungs are logged as skipped.

Installation requires package network access; inference downloads weights on
first use. Model downloads use the Hugging Face cache (`~/.cache/huggingface`
by default); set `HF_HOME` to override it and `HF_TOKEN` when authentication is
required. Results and server logs are written to `results/`.

Model downloads and runtime caches consume shared RAM too. Budgeting uses Linux
`MemAvailable` and leaves 16 GiB headroom. It does not clear OS caches or count
swap as GPU capacity. Keep the same runtime for comparable sweeps; the actual
vLLM version is recorded in metadata.

## llama.cpp

GGUF rungs and `llama_server_command()` are available, but automated GGUF sweeps
and a llama.cpp build are not provided by this project. The budget tool's
`--engine llama_cpp` view uses the same approximate weight accounting and runtime
allowance, not a calibrated GGUF allocator model.
