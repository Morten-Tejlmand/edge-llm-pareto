# edge-llm-pareto

Benchmark **quantization × concurrency on NVIDIA DGX Spark / GB10 with 128 GB
unified CPU/GPU memory**. The experiment measures how compression changes total
throughput, streaming speed and latency as concurrent requests increase.

## Revised study and vLLM

The [updated experiment plan](docs/study-plan.md) runs **llama.cpp on Unsloth
GGUF files and vLLM on official safetensors**, at five concurrency levels × 30
trials: 50 configurations and 1,500 measured trials (llama.cpp: 2 models × 3
GGUF representations; vLLM: Qwen BF16 / Qwen FP8 / NVIDIA NVFP4 and DeepSeek
BF16). The engines therefore run different weight files except at BF16; every
point is an engine-plus-artifact deployment, and the paper must say so
([why](docs/pilot-findings-vllm.md#decision-2026-11-17)).

The plan also adopts [Guldner et al.'s GSMM (2024)](https://doi.org/10.1016/j.future.2024.01.033)
as its measurement reference framework, with a [study-specific mapping](docs/study-plan.md#measurement-methodology-gsmm)
covering goals, metrics, procedure, setup, evaluation and baseline/idle measurements.

Add `--gsmm` to a sweep to record a system baseline before each variant launch
and server idle before each trial (30 seconds each, configurable). Results retain
raw reference traces, environment/sensor metadata and separately labeled adjusted
energy estimates. Gross energy remains the Pareto objective. Sensor validation
and pilot stabilization checks still need to be completed on the Spark.

The sweep now accepts `--engine vllm` with a separately provisioned
[GB10 vLLM runtime](docs/engines.md#vllm). `configs/study.yaml` supplies Qwen3.8-27B
and an explicitly older DeepSeek 8B pilot fallback: the latest DeepSeek V4.1
high-precision reference cannot fit on one Spark. Native vLLM and experimental
GGUF rows on vLLM stay configured only to reproduce the pilot; `scripts/qa_study.py` runs the study matrix. The commands below retain the original
llama.cpp smoke-test defaults.

## Run on the Spark

**One command.** Copy the repository to the Spark and, inside `tmux`, run:

```bash
scripts/spark_run.sh smoke   # setup + one quick trial per engine
scripts/spark_run.sh pilot   # + pilot subset: concurrency 1,4,16, 5 trials (~11 h estimate)
scripts/spark_run.sh full    # + full study: 50 configurations x 30 trials (re-estimate from pilot)
```

It runs `uv sync`, builds a CUDA `llama-server` if needed, checks the GPU,
downloads the pinned Unsloth GGUFs (vLLM fetches its pinned safetensors itself), runs a smoke test and then the chosen study,
and writes a report. vLLM is found automatically (`$VLLM`, `PATH`, `~/vllm-env`,
other `~/*/bin`); it is provisioned separately, see [engines](docs/engines.md#vllm).
Without it only llama.cpp runs. Results go to `results/pilot/` or `results/full/`;
re-running the same command after an interruption continues there and skips
completed trials. Logs go to `results/runs/`.

### Manual steps

The backend is **llama.cpp**, using GGUF models and a CUDA
`llama-server` binary. First [build llama-server](docs/engines.md#build-on-the-spark).
Then run from this repository:

```bash
uv sync
export PATH="$HOME/llama.cpp/build/bin:$PATH"
.venv/bin/python scripts/check_spark.py
.venv/bin/python scripts/budget.py

# Smoke test: Qwen3-1.7B Q4_K_M, concurrency 1, one request:
.venv/bin/python scripts/sweep.py

# Full GGUF sweeps, one model at a time:
.venv/bin/python scripts/sweep.py --model qwen3-1.7b --rung all --concurrency all --requests-per-agent 3
.venv/bin/python scripts/sweep.py --model qwen3-4b-2507 --rung all --concurrency all --requests-per-agent 3
.venv/bin/python scripts/report.py 'results/qwen3-1.7b-llama_cpp-*.jsonl'
```

Alternatively, pass `--llama-server /path/to/llama-server` to the runtime check
and sweep. GGUF weights download automatically. `uv sync` installs the Python
benchmark tools; the C++ server is built separately.

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

Performance requests default to a **256-output-token cap**, configurable with
`--output-tokens`; the study (`configs/study.yaml`) uses 2048. QA is uncapped by default within the configured context;
`--quality-max-tokens` sets its independent limit (`--max-tokens` for standalone
`evaluate.py`). Both paths retain actual output counts and termination reasons.
Context boundaries and existing request timeouts still apply.

## Memory defaults

- **70% memory planning budget** for llama.cpp slot estimates; no enforced process cap.
- **16 GiB additional host headroom** below current `MemAvailable`.
- **4 GiB estimated runtime overhead**, subtracted before estimating KV capacity.
- **4096 tokens per context**, default concurrency **1**, **one request**.
- Full sweep levels with `--concurrency all`: **1, 2, 4, 8, 16**.

The tighter of the utilization cap and available-memory-minus-headroom determines
the budget. Available memory is refreshed before each model launch. GB10 uses
`/proc/meminfo`, including reclaimable cache, rather than relying on dedicated
VRAM counters. Swap is not counted as capacity. These are starting settings,
**not a calibrated GB10 memory model or an OOM guarantee**.

Both budget and sweep accept `--gpu-mem-util`, `--host-reserve-gib`, and
`--runtime-overhead-gib`; use the same values for both. Longer contexts and higher
concurrency increase runtime requirements; compare estimates with server logs.
For llama.cpp these memory settings guide slot allocation only; they do not
enforce a process memory cap. KV uses f16 and each slot receives `--context` tokens.

## Models and measurements

| Base model | Configured GGUF variants |
|---|---:|
| Qwen3-1.7B | 7 |
| Qwen3-4B-Instruct-2507 | 5 |

The default command tests **Qwen3-1.7B GGUF Q4_K_M with one request at concurrency 1**.
Use `--rung all --concurrency all --requests-per-agent 3` for full sweeps: up to
42 points for 1.7B or 30 for 4B.
Unsupported variants are recorded as skipped.
The command exits with an error if no benchmark point succeeds. BF16 is included
for the 1.7B model. The llama.cpp backend downloads configured GGUF files
automatically and records the resolved model revision and server build.

Requests are synthetic long-prompt summarization workloads, not tool-using
agents. Recorded metrics include aggregate output tokens/s, streaming chunk
latency, time to first content, total request latency, and available GPU telemetry.
`per_agent_tok_s` is the reciprocal of mean content-chunk spacing; it excludes
initial queueing/prefill and is approximate when chunks contain multiple tokens.

The default report computes a speed-only Pareto front; `--energy` produces
energy–throughput or energy–latency frontiers with trial uncertainty and exports. Compare one base model and matching
workloads/runtime settings at a time. It excludes rows explicitly tagged throttled
or containing failed requests. Activity counters alone cannot establish a
memory-bandwidth/compute crossover. A fixed-task quality evaluator is available; representative evaluation data is still needed.

## Layout and validation

- `configs/models.yaml`: model/precision ladder.
- `scripts/budget.py`: memory estimates; `scripts/sweep.py`: server orchestration.
- `src/edge_llm_pareto/`: hardware probing, launch, benchmark and monitoring.
- `scripts/report.py`: tabular results and Pareto selection.
- [Pipeline](docs/pipeline.md), [open questions](docs/open-questions.md).

Local regression tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
The native runtime and actual model/kernel execution must be validated on
the Spark. No GB10 performance results are claimed by this setup.

## Research experiment pipeline

See [the experiment guide](docs/experiments.md) for repeated trials, warmup,
seeded ordering, resume, timestamped energy measurements, Jetson adapters,
quality evaluation, and energy Pareto plots. The default commands above remain
smoke tests; they do not enable a research measurement protocol automatically.


### Where the Python code lives

- `scripts/` contains terminal entry points: use `sweep.py` to run experiments,
  `report.py` to report results, and `evaluate.py` to score model answers.
- `src/edge_llm_pareto/` contains reusable functions imported by those commands.
  `bench.py` sends requests, `telemetry.py` collects hardware samples with one
  monitor, and `energy.py` integrates those samples into joules.

Workloads and request results are ordinary dictionaries. For example:

```python
from edge_llm_pareto.bench import make_workload, make_prompt

workload = make_workload(prompt_tokens=2000)
prompt = make_prompt(workload, seed=0)
print(workload["max_output_tokens"])  # 256
```

The monitor remains a class because it owns a background thread and must stop it
and release NVML when a run ends. Its samples are dictionaries too; GPU summaries
and energy calculations use the same samples.

Fixed-corpus research controls: [perplexity, fixed prompts, server slots, and combined-model reports](docs/experiments.md#fixed-corpus-study-perplexity-and-generation).
