#!/usr/bin/env bash
# One-shot runner for the DGX Spark (GB10): setup, model download, smoke test,
# then the selected study run and report. Safe to re-run: every step reuses
# what already exists (venv, llama.cpp build, Hugging Face cache), and an
# interrupted pilot/full run continues from results/<mode>/, skipping
# completed trials. vLLM is found automatically ($VLLM, PATH, ~/vllm-env, ...).
#
# Usage (from the repository root, ideally inside tmux):
#   scripts/spark_run.sh smoke   # setup + one quick trial per engine (~30 min)
#   scripts/spark_run.sh pilot   # smoke + pilot subset (~11 h estimate)
#   scripts/spark_run.sh full    # smoke + full study, 1,500 trials (re-estimate from pilot)
#
# Optional environment variables:
#   LLAMA_SERVER=/path/to/llama-server   use an existing build instead of building
#   LLAMA_CPP_DIR=$HOME/llama.cpp        where to clone/build llama.cpp
#   VLLM=/path/to/vllm                   only needed if vLLM is somewhere unusual;
#                                        if none is found, only llama.cpp is run
#   HF_TOKEN=...                         Hugging Face token, if downloads need one
#   ENGINES=llama_cpp                    run llama.cpp only, even if vLLM is installed
#   SKIP_SMOKE=1                         skip the smoke test (not recommended)
set -euo pipefail

MODE="${1:-pilot}"
case "$MODE" in smoke|pilot|full) ;; *)
  echo "Usage: $0 [smoke|pilot|full]" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$ROOT/results/runs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/spark_run-$MODE-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "Mode: $MODE  (log: $LOG)"
[ -n "${TMUX:-}${STY:-}" ] || echo "Note: not inside tmux/screen; a dropped SSH session will stop the run."

# ---------------------------------------------------------------- preflight
step "Checking the machine"
[ "$(uname -m)" = "aarch64" ] || echo "Warning: expected aarch64 (GB10); found $(uname -m)."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: NVIDIA driver missing."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || die "nvidia-smi failed."
command -v uv >/dev/null || die "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
FREE_GB=$(df -Pk "$HF_CACHE" | awk 'NR==2 {print int($4/1024/1024)}')
echo "Free disk at $HF_CACHE: ${FREE_GB} GB (models need ~140 GB)"
[ "$FREE_GB" -ge 150 ] || echo "Warning: less than 150 GB free; downloads already in the cache do not count again."

# ---------------------------------------------------------------- python env
step "Python environment (uv sync)"
if [ -e .venv ] && ! .venv/bin/python -c 'import sys' 2>/dev/null; then
  echo "Existing .venv is not runnable here (copied from another machine?); recreating."
  rm -rf .venv
fi
uv sync
PY="$ROOT/.venv/bin/python"

# ---------------------------------------------------------------- llama.cpp
step "llama.cpp server"
if [ -n "${LLAMA_SERVER:-}" ]; then
  [ -x "$LLAMA_SERVER" ] || die "LLAMA_SERVER=$LLAMA_SERVER is not executable."
else
  LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
  LLAMA_SERVER="$LLAMA_CPP_DIR/build/bin/llama-server"
  if [ ! -x "$LLAMA_SERVER" ]; then
    for tool in git cmake c++; do command -v "$tool" >/dev/null || die "$tool is required to build llama.cpp."; done
    NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
    [ -x "$NVCC" ] || die "CUDA compiler not found at $NVCC (set NVCC=...)."
    [ -d "$LLAMA_CPP_DIR/.git" ] || git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$(dirname "$(dirname "$NVCC")")" \
      -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server -j "$(nproc)"
  fi
  echo "llama.cpp commit: $(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
"$LLAMA_SERVER" --version || true
"$LLAMA_SERVER" --list-devices 2>&1 | tee /dev/stderr | grep CUDA0 >/dev/null \
  || die "llama-server does not list CUDA0: it was not built with CUDA for this GPU."
"$PY" scripts/check_spark.py --llama-server "$LLAMA_SERVER"

# ---------------------------------------------------------------- vLLM
step "vLLM runtime"
ENGINES="${ENGINES:-llama_cpp,vllm}"
find_vllm() {  # first working vllm: $VLLM, PATH, then common virtualenv locations
  local candidate
  for candidate in "${VLLM:-}" "$(command -v vllm 2>/dev/null || true)" \
      "$HOME/vllm-env/bin/vllm" "$HOME/.vllm-env/bin/vllm" "$HOME/vllm/.venv/bin/vllm" \
      "$HOME"/*/bin/vllm "$HOME"/*/.venv/bin/vllm "$HOME"/.venv*/bin/vllm /opt/*/bin/vllm; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    [ "$candidate" -ef "$ROOT/.venv/bin/vllm" ] && continue   # never the project env
    if "$candidate" --version >/dev/null 2>&1; then echo "$candidate"; return 0; fi
  done
  return 1
}
if [ "$ENGINES" = "llama_cpp" ]; then
  echo "ENGINES=llama_cpp: vLLM is skipped for this run."
  VLLM="vllm"
elif VLLM="$(find_vllm)"; then
  echo "vLLM: $VLLM ($("$VLLM" --version 2>&1 | tail -1))"
  # vLLM compiles kernels at startup (FlashInfer, torch inductor): it needs
  # ninja and the CUDA compiler on PATH, and CUDA_HOME. Without them it fails
  # with "No such file or directory: 'ninja'" or silently disables FlashInfer.
  VLLM_BIN_DIR="$(dirname "$VLLM")"
  CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "${NVCC:-/usr/local/cuda/bin/nvcc}")")}"
  [ -x "$CUDA_HOME/bin/nvcc" ] || echo "WARNING: no nvcc at $CUDA_HOME/bin; set CUDA_HOME. vLLM kernel compilation may fail."
  export CUDA_HOME PATH="$PATH:$CUDA_HOME/bin:$VLLM_BIN_DIR"
  if ! command -v ninja >/dev/null; then
    echo "Installing ninja into $(dirname "$VLLM_BIN_DIR") (needed by vLLM)"
    uv pip install --python "$VLLM_BIN_DIR/python" ninja \
      || die "Could not install ninja; run: uv pip install --python $VLLM_BIN_DIR/python ninja"
  fi
  echo "ninja: $(command -v ninja)  nvcc: $(command -v nvcc || echo missing)  CUDA_HOME=$CUDA_HOME"
  # Prebuilt kernels avoid slow, variable compilation at every server start.
  KERNEL_MARK="$HOME/.cache/edge-llm-pareto/vllm-kernels-$("$VLLM" --version 2>&1 | tail -1)"
  if [ ! -e "$KERNEL_MARK" ]; then
    if "$VLLM" download-kernels; then mkdir -p "$(dirname "$KERNEL_MARK")"; touch "$KERNEL_MARK";
    else echo "Note: 'vllm download-kernels' failed; vLLM will compile kernels at startup instead."; fi
  fi
else
  echo "WARNING: no working vLLM found (searched \$VLLM, PATH, ~/vllm-env and other ~/*/bin)."
  echo "         Set it up with NVIDIA's Spark vLLM playbook;"
  echo "         see docs/engines.md. Continuing with llama.cpp only."
  ENGINES="llama_cpp"
  VLLM="vllm"
fi

# ---------------------------------------------------------------- downloads
step "Downloading pinned Unsloth GGUF files (reuses the Hugging Face cache)"
"$PY" - <<'EOF'
import sys, yaml
sys.path.insert(0, 'src')
from edge_llm_pareto import serve
cfg = yaml.safe_load(open('configs/study.yaml'))
for model, rungs in cfg['study']['model_rungs'].items():
    engine = cfg['models'][model]['engine_rungs']['llama_cpp']
    files = {r['rung']: r['file'] for r in engine['rungs']}
    for rung in rungs:
        result = serve.download_gguf(engine['repo'], files[rung], engine['revision'])
        print(f"  ok  {model} {rung} @ {result['revision'][:12]}: {', '.join(result['files'])}", flush=True)
EOF

COMMON=(--engines "$ENGINES" --llama-server "$LLAMA_SERVER" --vllm "$VLLM")

run_study() {  # $1 = results dir, rest = extra qa_study arguments
  local dir="$1"; shift
  "$PY" scripts/qa_study.py "${COMMON[@]}" --results-dir "$dir" "$@" --dry-run | sed -n '/^Resuming/p; /^Planned/p'
  "$PY" scripts/qa_study.py "${COMMON[@]}" --results-dir "$dir" "$@"
}

report() {  # $1 = results dir
  step "Report for $1"
  local extra=()
  if ls "$1"/*.likelihood.json >/dev/null 2>&1; then
    if "$PY" scripts/likelihood_report.py "$1/*.likelihood.json" --output "$1/report/likelihood.json"; then
      extra=(--likelihood-results "$1/report/likelihood.json")
    else
      echo "Likelihood summary failed; the energy report continues without QA perplexity/KL."
    fi
  fi
  "$PY" scripts/report.py "$1/*.jsonl" --energy --export-dir "$1/report" "${extra[@]}" \
    || echo "Report failed or found no valid rows; inspect $1 and the server logs there."
}

skipped_rows() {  # count skipped/failed rows in a results dir
  "$PY" - "$1" <<'EOF'
import glob, json, sys
n = 0
for path in glob.glob(f'{sys.argv[1]}/*.jsonl'):
    for line in open(path):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('skipped') or row.get('failed'):
            n += 1
            reason = (f"skipped: {row.get('reason')}" if row.get('skipped')
                      else f"{row['failed']} failed requests (c={row.get('concurrency')}, trial={row.get('trial')})")
            print(f"  {row.get('engine')} {row.get('label')}: {reason}", file=sys.stderr)
print(n)
EOF
}

# ---------------------------------------------------------------- smoke
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  step "Smoke test: DeepSeek 8B (llama.cpp UD-Q4_K_XL, vLLM BF16), concurrency 1, one trial"
  SMOKE_DIR="results/smoke-$STAMP"
  run_study "$SMOKE_DIR" --models deepseek-r1-0528-qwen3-8b --rungs gguf-ud-q4_k_xl,native-bf16 \
    --concurrency 1 --repeats 1 || die "Smoke test failed; see $SMOKE_DIR and $LOG."
  BAD=$(skipped_rows "$SMOKE_DIR")
  [ "$BAD" = "0" ] || die "Smoke test recorded $BAD skipped/failed rows; see $SMOKE_DIR."
  echo "Smoke test passed."
fi
[ "$MODE" = "smoke" ] && { step "Done (smoke only)."; exit 0; }

# ---------------------------------------------------------------- main run
# Fixed folder per mode: re-running the script continues an interrupted run
# (completed trials are skipped). To start over, rename or delete the folder.
OUT="results/$MODE"
if ls "$OUT"/*.jsonl >/dev/null 2>&1; then
  echo "Existing results in $OUT: resuming; completed trials will be skipped."
fi
if [ "$MODE" = "pilot" ]; then
  step "Pilot: all models and quantizations, concurrency 1,4,16, 5 trials (~11 h estimate)"
  run_study "$OUT" --concurrency 1,4,16 --repeats 5 --resume || echo "Some pilot cells failed; continuing to the report."
else
  step "Full study: 50 configurations x 30 trials"
  run_study "$OUT" --resume || echo "Some study cells failed; continuing to the report."
fi

BAD=$(skipped_rows "$OUT")
echo "Skipped/failed rows (including attempts before a resume): $BAD"
report "$OUT"
step "Finished. Results: $OUT   Log: $LOG"
