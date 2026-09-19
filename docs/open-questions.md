# Remaining experiment work

1. **GB10 validation:** install and smoke-test the native runtime on the actual Spark;
   validate every configured quantization kernel. Successful installation is not
   evidence that a particular model/kernel combination has been tested.
2. **Memory calibration:** the 4 GiB runtime allowance is a starting estimate.
   Profile activations, CUDA graphs and quantization metadata at the chosen
   context and concurrency. CPU model loading competes for the same RAM. Linux
   MemAvailable can differ from CUDA's view; startup remains authoritative.
3. **Accuracy:** the Pareto front ranks speed only. Add a common quality task
   before choosing a precision variant for real agent workloads.
4. **Benchmark protocol:** add warmup, repeated measurements and uncertainty.
   Streaming chunks are not always single tokens; per-agent decode speed does
   not include time spent waiting for the first content chunk.
5. **Mechanism:** NVML utilization is an activity signal, not a roofline analysis.
   Use suitable profiling to establish compute versus bandwidth saturation.
6. **Workload scale:** current 1.7B and 4B models are useful baselines. Longer
   contexts, more concurrency or larger models may better exercise 128 GB.
7. **Engine coverage:** llama.cpp launch helpers and GGUF configuration exist,
   but automated sweeps are still vLLM-only. Dynamic GGUF and bitsandbytes formats
   cannot be assumed quality-equivalent.
8. **Comparability:** compare the same base model, workload, hardware and runtime.
   Preflight throttle tags exclude bad rows; clocks are sampled for inspection,
   but no complete mid-run thermal/power validity detector is implemented.
