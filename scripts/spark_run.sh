#!/usr/bin/env bash
# One-shot runner for the DGX Spark (GB10): setup, model download, smoke test,
# then the selected study run and report. Safe to re-run: every step reuses
# what already exists (venv, llama.cpp build, Hugging Face cache).
#
# Usage (from the repository root, ideally inside tmux):
#   scripts/spark_run.sh smoke   # setup + one quick trial per engine (~30 min)
#   scripts/spark_run.sh pilot   # smoke + pilot subset (~11 h estimate)
#   scripts/spark_run.sh full    # smoke + full study, 1,800 trials (~4 days estimate)
#
# Optional environment variables:
#   LLAMA_SERVER=/path/to/llama-server   use an existing build instead of building
#   LLAMA_CPP_DIR=$HOME/llama.cpp        where to clone/build llama.cpp
#   VLLM=/path/to/vllm                   vLLM executable (with vllm-gguf-plugin);
#                                        if absent, only llama.cpp is run
#   HF_TOKEN=...                         Hugging Face token, if downloads need one
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
ENGINES="llama_cpp,vllm"
VLLM="${VLLM:-$(command -v vllm || true)}"
if [ -n "$VLLM" ] && "$VLLM" --version >/dev/null 2>&1; then
  echo "vLLM: $VLLM ($("$VLLM" --version 2>&1 | tail -1))"
else
  echo "WARNING: no working vLLM found (set VLLM=/path/to/vllm)."
  echo "         Set it up with NVIDIA's Spark vLLM playbook plus vllm-gguf-plugin;"
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
  "$PY" scripts/qa_study.py "${COMMON[@]}" --results-dir "$dir" "$@" --dry-run | sed -n 1p
  "$PY" scripts/qa_study.py "${COMMON[@]}" --results-dir "$dir" "$@"
}

report() {  # $1 = results dir
  step "Report for $1"
  "$PY" scripts/report.py "$1/*.jsonl" --energy --export-dir "$1/report" \
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
  step "Smoke test: DeepSeek 8B UD-Q4_K_XL, concurrency 1, one trial per engine"
  SMOKE_DIR="results/smoke-$STAMP"
  run_study "$SMOKE_DIR" --models deepseek-r1-0528-qwen3-8b --rungs gguf-ud-q4_k_xl \
    --concurrency 1 --repeats 1 || die "Smoke test failed; see $SMOKE_DIR and $LOG."
  BAD=$(skipped_rows "$SMOKE_DIR")
  [ "$BAD" = "0" ] || die "Smoke test recorded $BAD skipped/failed rows; see $SMOKE_DIR."
  echo "Smoke test passed."
fi
[ "$MODE" = "smoke" ] && { step "Done (smoke only)."; exit 0; }

# ---------------------------------------------------------------- main run
if [ "$MODE" = "pilot" ]; then
  OUT="results/pilot-$STAMP"
  step "Pilot: all models and quantizations, concurrency 1,4,16, 5 trials (~11 h estimate)"
  run_study "$OUT" --concurrency 1,4,16 --repeats 5 || echo "Some pilot cells failed; continuing to the report."
else
  OUT="results/full-$STAMP"
  step "Full study: 60 configurations x 30 trials (~4 days estimate)"
  run_study "$OUT" || echo "Some study cells failed; continuing to the report."
fi

BAD=$(skipped_rows "$OUT")
echo "Skipped/failed rows: $BAD"
report "$OUT"
step "Finished. Results: $OUT   Log: $LOG"
