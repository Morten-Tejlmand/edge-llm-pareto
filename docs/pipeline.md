# Pipeline on GB10

1. **Check runtime** with `scripts/check_spark.py` in the Spark's native Python environment.
   It verifies GB10 and runs a small CUDA operation.
2. **Estimate capacity** with `scripts/budget.py`: fetch model configs, separate
   transformer and embedding weights, then calculate KV bytes per token.
3. **Probe shared memory** using Linux `MemTotal` and `MemAvailable`. The budget
   is `min(total × utilization, available − host reserve) − weights − overhead`.
   Defaults are 0.7 utilization, 16 GiB host reserve, 4 GiB runtime estimate.
4. **Preflight** power/thermal conditions. Missing telemetry does not prevent
   execution; explicit throttle conditions require `--allow-throttled`.
5. **Per precision variant**, refresh available memory, skip if one full context
   cannot fit, launch vLLM with the effective memory fraction, wait for health.
   Prefix caching is disabled. Actual runtime allocation can differ from estimates.
6. **Benchmark** concurrent streaming requests while sampling GPU telemetry.
   Unsupported counters do not suppress unrelated measurements. Shared-system
   memory is labelled separately from GPU framebuffer memory.
7. **Persist** each result including estimated KV ceiling, hardware, effective
   memory fraction and preflight throttle status; stop the server process group.
8. **Report** matching runs for one base model. Failed and explicitly throttled
   rows are excluded from the speed-only Pareto front. Missing telemetry is not
   evidence that hardware operated without throttling.

No warmup/repeated-trial statistical protocol or accuracy evaluation is currently
implemented. The fixed request pool also drains at the end of each measurement,
so reported throughput includes startup/tail effects. See open questions.
