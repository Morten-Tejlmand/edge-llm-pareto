# edge-llm-pareto

Benchmark **quantization × concurrency on NVIDIA DGX Spark / GB10 with 128 GB
unified CPU/GPU memory**. The experiment measures how compression changes total
throughput, streaming speed and latency as concurrent requests increase.

## Run on the Spark

Run from the repository root using the native Python environment. See
[engine setup](docs/engines.md) for setup and GB10 validation requirements.

```bash
uv sync
.venv/bin/python scripts/check_spark.py
.venv/bin/python scripts/budget.py

# Smoke test: one model, BF16, concurrency 1, one request:
.venv/bin/python scripts/sweep.py

# Full vLLM sweeps, one model at a time:
.venv/bin/python scripts/sweep.py --model qwen3-1.7b --rung all --concurrency all --requests-per-agent 3
.venv/bin/python scripts/sweep.py --model qwen3-4b-2507 --rung all --concurrency all --requests-per-agent 3
.venv/bin/python scripts/report.py 'results/qwen3-1.7b-vllm-*.jsonl'
```

Results and server logs are written to `results/`. Model downloads use the
Hugging Face cache; set `HF_TOKEN` when authentication is required.

To estimate the Spark's nominal capacity from a development machine:

```bash
.venv/bin/python scripts/budget.py --target gb10
```

This uses a **nominal 128 GiB planning assumption**, not the development machine's
GPU. It downloads only model configuration JSON. Real runs always probe the GB10
and use OS-reported physical memory, which can be smaller than nominal capacity.
The sweep refuses a different GPU to avoid accidentally collecting laptop data.

## Memory defaults

- **70% process memory utilization**, passed explicitly to vLLM.
- **16 GiB additional host headroom** below current `MemAvailable`.
- **4 GiB estimated runtime overhead**, subtracted before estimating KV capacity.
- **4096 tokens per context**, default concurrency **1**, **one request**.
- Full sweep levels with `--concurrency all`: **1, 2, 4, 8, 16, 32**.

The tighter of the utilization cap and available-memory-minus-headroom determines
the budget. Available memory is refreshed before each model launch. GB10 uses
`/proc/meminfo`, including reclaimable cache, rather than relying on dedicated
VRAM counters. Swap is not counted as capacity. These are starting settings,
**not a calibrated GB10 memory model or an OOM guarantee**.

Both budget and sweep accept `--gpu-mem-util`, `--host-reserve-gib`, and
`--runtime-overhead-gib`; use the same values for both. Longer contexts and higher
concurrency increase runtime requirements; compare estimates with server logs.

## Models and measurements

| Base model | vLLM precision variants | Configured GGUF variants |
|---|---:|---:|
| Qwen3-1.7B | 7 | 7 |
| Qwen3-4B-Instruct-2507 | 4 | 2 |

The default command tests **Qwen3-1.7B BF16 with one request at concurrency 1**.
Use `--rung all --concurrency all --requests-per-agent 3` for full sweeps: up to
42 points for 1.7B or 24 for 4B. Unsupported variants are recorded as skipped.
The command exits with an error if no benchmark point succeeds. BF16 is included
for both models. GGUF configurations and launcher helpers exist, but automated
llama.cpp sweeps remain unimplemented.

Requests are synthetic long-prompt summarization workloads, not tool-using
agents. Recorded metrics include aggregate output tokens/s, streaming chunk
latency, time to first content, total request latency, and available GPU telemetry.
`per_agent_tok_s` is the reciprocal of mean content-chunk spacing; it excludes
initial queueing/prefill and is approximate when chunks contain multiple tokens.

The report computes a speed-only Pareto front. Compare one base model and matching
workloads/runtime settings at a time. It excludes rows explicitly tagged throttled
or containing failed requests. Activity counters alone cannot establish a
memory-bandwidth/compute crossover. Accuracy evaluation is still needed.

## Layout and validation

- `configs/models.yaml`: model/precision ladder.
- `scripts/budget.py`: memory estimates; `scripts/sweep.py`: server orchestration.
- `src/edge_llm_pareto/`: hardware probing, launch, benchmark and monitoring.
- `scripts/report.py`: tabular results and Pareto selection.
- [Pipeline](docs/pipeline.md), [open questions](docs/open-questions.md).

Local regression tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
The native runtime and actual model/kernel execution must be validated on
the Spark. No GB10 performance results are claimed by this setup.
