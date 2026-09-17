# XPU inference packaging and recovery

Source base: `3d85f3c988ba2c6e809482aeb901ffc89b1defea`.

Build the tested CPU-FBGEMM frontend package with:

```bash
TORCHREC_USE_FBGEMM_CPU=1 python -m pip wheel --no-build-isolation --no-deps .
```

This changes only wheel dependency metadata from `fbgemm-gpu` to
`fbgemm-gpu-cpu`. The default build remains unchanged. The consuming environment
pins `fbgemm-gpu-cpu==1.8.0` and `torch==2.13.0+xpu`, and installs `fbgemm-xpu`.
Do not install both FBGEMM distributions into the same environment.

The CPU sparse baseline and non-sharded INT4/INT8 `EmbeddingCollection` on XPU
passed functional tests on DUT1013PVC. Bounds v1 is enabled and its actual XPU
invocation is observed in the plugin and integrated model tests.

No distributed sharder/planner changes are needed for this non-DMP path.
`create_infer_embedding_sharding` still rejects XPU storage and the planner has
no XPU QUANT bandwidth entry. Sharded inference requires a separate tested port,
placement/routing validation, capacity planning, and a collective hardware gate.
This branch does not claim TW/RW/CW or multi-device readiness.

Reproduction scripts and measured results:
`https://github.com/mkrze/generative-recommenders/tree/xpu/mlperf-dlrmv3-inference/scripts/mlperf-dlrm-v3-xpu`.