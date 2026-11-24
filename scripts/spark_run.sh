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
#   scripts/spark_run.sh vllm-check  # vLLM only: every vLLM row once (DeepSeek first),
#                                    # then tokenizer check, likelihood and QA summary
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
case "$MODE" in smoke|pilot|full|vllm-check) ;; *)
  echo "Usage: $0 [smoke|pilot|full|vllm-check]" >&2; exit 2;; esac

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

# Leftover inference servers from an interrupted run hold most of the unified
# memory, and the next run then fails with "predicted not to fit" or "server
# slots exceed estimated capacity". Stop our own leftovers; refuse to touch a
# run that is still in progress or a port held by something else.
BENCH_PORT="${BENCH_PORT:-8000}"
# Match the program actually being run (argv[0..2]), not any command line that
# merely mentions these names (an editor, `tail -f`, a shell).
own_procs() { ps -u "$(id -u)" -o pid=,args= | awk -v me=$$ "$1"; }
find_servers() { own_procs '$1 != me && ($2 ~ /(^|\/)llama-server$/ || $2 ~ /^VLLM::/ ||
  ($2 ~ /(^|\/)vllm$/ && $3 == "serve") || ($3 ~ /(^|\/)vllm$/ && $4 == "serve")) {print $1}'; }
ACTIVE=$(own_procs '$2 ~ /(^|\/)python[0-9.]*$/ && $3 ~ /scripts\/(sweep|qa_study)\.py$/ {print $1}')
[ -z "$ACTIVE" ] || die "Another benchmark run is in progress (PIDs: $(echo $ACTIVE)). Check 'tmux ls'; stop it before starting a new one."
STALE=$(find_servers)
if [ -n "$STALE" ]; then
  echo "Stopping leftover inference servers from an earlier run:"
  ps -o pid=,etime=,args= -p "$(echo $STALE | tr ' ' ,)" | cut -c1-160
  kill -INT $STALE 2>/dev/null || true
  for _ in $(seq 30); do
    STALE=$(find_servers)
    [ -z "$STALE" ] && break
    sleep 1
  done
  if [ -n "$STALE" ]; then kill -KILL $STALE 2>/dev/null || true; sleep 3; fi
  STALE=$(find_servers)
  [ -z "$STALE" ] || die "Could not stop leftover servers (PIDs: $(echo $STALE))."
  echo "Leftover servers stopped."
fi
if command -v ss >/dev/null && ss -ltn "sport = :$BENCH_PORT" | grep -q LISTEN; then
  die "Port $BENCH_PORT is in use by another process: $(ss -ltnp "sport = :$BENCH_PORT" | tail -n +2). Stop it first."
fi
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
echo "Available memory: ${AVAIL_GB} GB"
[ "$AVAIL_GB" -ge 100 ] || echo "Warning: under 100 GB available; check 'nvidia-smi' for other GPU processes. Servers may be predicted not to fit."

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
if [ "$MODE" = "vllm-check" ]; then
  echo "vllm-check: llama.cpp is not used."
  LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
elif [ -n "${LLAMA_SERVER:-}" ]; then
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
if [ "$MODE" != "vllm-check" ]; then
  "$LLAMA_SERVER" --version || true
  "$LLAMA_SERVER" --list-devices 2>&1 | tee /dev/stderr | grep CUDA0 >/dev/null \
    || die "llama-server does not list CUDA0: it was not built with CUDA for this GPU."
  "$PY" scripts/check_spark.py --llama-server "$LLAMA_SERVER"
fi

# ---------------------------------------------------------------- vLLM
step "vLLM runtime"
ENGINES="${ENGINES:-llama_cpp,vllm}"
[ "$MODE" = "vllm-check" ] && ENGINES="vllm"
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
  [ "$MODE" = "vllm-check" ] && die "vllm-check needs a working vLLM (set VLLM=/path/to/vllm)."
  ENGINES="llama_cpp"
  VLLM="vllm"
fi

# ---------------------------------------------------------------- downloads
if [ "$MODE" != "vllm-check" ]; then
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
fi
if [[ ",$ENGINES," == *",vllm,"* ]]; then
step "Downloading pinned vLLM safetensors (reuses the Hugging Face cache)"
# vLLM would otherwise download at server start, inside its startup timeout
# (Qwen3.8-27B BF16 is 52 GB: 7 of its 10 minutes in the pilot).
"$PY" - <<'EOF'
import yaml
from huggingface_hub import snapshot_download
cfg = yaml.safe_load(open('configs/study.yaml'))
for model, rungs in cfg['study']['native_vllm_rungs'].items():
    by_name = {r['rung']: r for r in cfg['models'][model]['engine_rungs']['vllm']['rungs']}
    for rung in rungs:
        r = by_name[rung]
        path = snapshot_download(r['repo'], revision=r['revision'],
                                 allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])
        print(f"  ok  {model} {rung} @ {r['revision'][:12]}: {path}", flush=True)
EOF
fi

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
        if row.get('skipped') or row.get('failed') or row.get('token_accounting_error'):
            n += 1
            reason = (f"skipped: {row.get('reason')}" if row.get('skipped')
                      else row.get('token_accounting_error') or f"{row['failed']} failed requests (c={row.get('concurrency')}, trial={row.get('trial')})")
            print(f"  {row.get('engine')} {row.get('label')}: {reason}", file=sys.stderr)
print(n)
EOF
}

validate_answers() {  # count trials whose answers show a decoding or extraction failure
  # Catches the two pilot bugs that passed the smoke test: raw 'Ġ'/'Ċ' tokenizer
  # symbols (DeepSeek on vLLM) and reasoning left in the scored answer (Qwen on
  # vLLM: the chat template opens <think> in the prompt). Exact match 0 is only
  # a warning: one c=1 trial scores just 3 questions.
  "$PY" - "$1" <<'EOF'
import glob, json, sys
bad = 0
for path in sorted(glob.glob(f'{sys.argv[1]}/*.jsonl')):
    for line in open(path):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('scenario') != 'generation_load' or not row.get('request_results'):
            continue
        name = f"{row.get('base_model')} {row.get('engine')} {row.get('label')} c={row.get('concurrency')}"
        ok = [r for r in row['request_results'] if r.get('ok')]
        symbols = sum(any(s in (r.get('raw_answer') or '') for s in ('Ġ', 'Ċ')) for r in ok)
        leaked = sum('think>' in (r.get('answer') or '') for r in ok)
        if symbols or leaked:
            bad += 1
            print(f"  ANSWER CHECK FAILED {name}: {symbols} answers with raw tokenizer symbols, "
                  f"{leaked} with reasoning tags left in the scored answer", file=sys.stderr)
            example = next((r for r in ok if 'think>' in (r.get('answer') or '') or 'Ġ' in (r.get('raw_answer') or '')), None)
            if example:
                print(f"    e.g. answer: {(example.get('answer') or '')[-120:]!r}", file=sys.stderr)
        elif ok and row.get('qa_exact_match') == 0:
            print(f"  warning: {name} scored exact match 0 on {len(ok)} answers; inspect them", file=sys.stderr)
print(bad)
EOF
}

# ---------------------------------------------------------------- vLLM check
if [ "$MODE" = "vllm-check" ]; then
  OUT="results/vllm-check-$STAMP"
  # DeepSeek first: small, and the model with the known vLLM tokenizer problem.
  for model in deepseek-r1-0528-qwen3-8b qwen3.8-27b; do
    step "vLLM check: $model, every vLLM row, concurrency 1, one trial"
    run_study "$OUT" --models "$model" --concurrency 1 --repeats 1 \
      || echo "Some $model cells failed; continuing."
  done
  step "vLLM check summary"
  CHECK_BAD=$(skipped_rows "$OUT")
  echo "Skipped/failed rows: $CHECK_BAD"
  ANSWER_BAD=$(validate_answers "$OUT")
  echo "Trials failing the answer check: $ANSWER_BAD"
  "$PY" scripts/likelihood_report.py "$OUT/*.likelihood.json" --output "$OUT/report/likelihood.json" \
    || echo "No likelihood results; see the sweep output above."
  "$PY" - "$OUT" <<'EOF'
import glob, json, sys
print(f"\n{'model':<28}{'rung':<16}{'QA EM':>7}{'hit cap':>9}  first answer (raw, decoding check)")
for path in sorted(glob.glob(f'{sys.argv[1]}/*.jsonl')):
    for line in open(path):
        row = json.loads(line)
        if row.get('skipped'):
            print(f"{row.get('model', '?'):<28}{row['label']:<16}  SKIPPED: {row.get('reason')}")
            continue
        if row.get('scenario') or 'qa_exact_match' not in row:
            continue
        answer = next((r.get('raw_answer') or '' for r in row['request_results'] if r.get('ok')), '')
        print(f"{row['base_model']:<28}{row['label']:<16}{row['qa_exact_match']:>7.2f}"
              f"{row['length_limited_requests']:>4}/{row['requests']:<4}  {answer[-70:]!r}")
print("\nLook for: tokenizer encode/decode OK, no 'Ġ'/'Ċ' symbols in answers, and KL to")
print("vLLM BF16 above 0 for qwen-fp8 / nvidia-nvfp4. One trial at c=1 is not a quality estimate.")
EOF
  if [ "$CHECK_BAD" != "0" ] || [ "$ANSWER_BAD" != "0" ]; then
    die "vLLM check failed ($CHECK_BAD skipped/failed rows, $ANSWER_BAD trials failing the answer check); see $OUT."
  fi
  step "Done: every vLLM row passed. Results: $OUT   Log: $LOG"
  exit 0
fi

# ---------------------------------------------------------------- smoke
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  step "Smoke test: DeepSeek 8B (llama.cpp UD-Q4_K_XL, vLLM BF16), concurrency 1, one trial"
  SMOKE_DIR="results/smoke-$STAMP"
  run_study "$SMOKE_DIR" --models deepseek-r1-0528-qwen3-8b --rungs gguf-ud-q4_k_xl,native-bf16 \
    --concurrency 1 --repeats 1 || die "Smoke test failed; see $SMOKE_DIR and $LOG."
  BAD=$(skipped_rows "$SMOKE_DIR")
  [ "$BAD" = "0" ] || die "Smoke test recorded $BAD skipped/failed rows; see $SMOKE_DIR."
  BAD=$(validate_answers "$SMOKE_DIR")
  [ "$BAD" = "0" ] || die "Smoke test: $BAD trials failed the answer check; see $SMOKE_DIR."
  echo "Smoke test passed."
fi
[ "$MODE" = "smoke" ] && { step "Done (smoke only)."; exit 0; }

# ---------------------------------------------------------------- preflight
# Before the multi-day study: every configuration once at c=1 (both engines,
# both models), with the tokenizer and answer checks. The smoke test covers only
# DeepSeek, which let the Qwen-on-vLLM scoring bug through the pilot.
if [ "$MODE" = "full" ] && [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  step "Preflight: every configuration once at concurrency 1 (~1.5-2 h; SKIP_PREFLIGHT=1 skips)"
  PRE_DIR="results/preflight-$STAMP"
  run_study "$PRE_DIR" --concurrency 1 --repeats 1 || echo "Some preflight cells failed."
  BAD=$(skipped_rows "$PRE_DIR")
  [ "$BAD" = "0" ] || die "Preflight recorded $BAD skipped/failed rows; see $PRE_DIR. Fix them before the full study."
  BAD=$(validate_answers "$PRE_DIR")
  [ "$BAD" = "0" ] || die "Preflight: $BAD trials failed the answer check; see $PRE_DIR."
  echo "Preflight passed: every configuration starts, tokenizes and scores correctly."
fi

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
