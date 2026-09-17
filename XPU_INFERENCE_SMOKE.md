# XPU inference packaging and recovery

## New experiment: single-XPU table-wise QUANT

Branch: `xpu/quant-inference-sharding-experiment`, based on the validated
packaging/inference commit `b3642a34e58c3424d3ac89bd3a5ab072ebc76a34`.
This is an isolated follow-up, not an extension of the 151/150/#127 PR branches
or the `mlperf-dlrmv3-xpu-smoke-2026-09-17` recovery tag.

On DUT1013PVC, INT4/INT8 and D=16/512 passed real
`EmbeddingShardingPlanner.plan`, table-wise sharding through `_shard_modules`,
CPU and non-sharded XPU output parity, and serialized state restoration after
deliberately zeroing a fresh module's packed weights. Two tables (17/23 rows),
three requests, empty segments, one `xpu:0`, no process group or collectives.
The combined experiment and existing bandwidth tests passed: six tests and
seven subtests. Bounds v1 and IntNBit lookup were observed on XPU.

Required configuration:
- Set `TORCHREC_XPU_TESTS=1`; missing XPU is an error in the opted-in tests.
- Import `fbgemm_xpu` before TorchRec. Keep the validated Torch 2.13.0+xpu,
	FBGEMM CPU 1.8.0 and plugin commit `a927f5c`/wheel from the GR manifest.
- Use `set_propogate_device(True)` (the existing API's spelling) and restore
	the prior setting when finished; otherwise inference lookup defaults to CUDA.
- Explicit table-wise/QUANT constraints, local `ShardingEnv` world_size=1,
	XPU DEVICE storage and row alignment 16.
- Use `quant_state_dict_split_scale_bias=True` consistently. For `torch.save`,
	clone each state value to separate the differently typed views of packed
	storage; directly saving the aliasing state dictionary fails in this stack.
	The test loads only its own in-memory checkpoint with `weights_only=False`;
	this is not a recommendation to unpickle untrusted checkpoints.

Run from the experimental TorchRec checkout using the prepared XPU venv:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCHREC_XPU_TESTS=1 OMP_NUM_THREADS=4
python -c 'import fbgemm_xpu, pytest, sys; sys.exit(pytest.main(sys.argv[1:]))' \
	torchrec/distributed/tests/test_quant_sequence_xpu.py \
	torchrec/distributed/planner/tests/test_constants.py -q -s
```

Remaining limitations: XPU RW/CW and world_size>1 are explicitly rejected by
this experiment. `fbgemm::all_to_one_device` still lacks XPU support. The sole
output already on the exact destination XPU takes an identity path, not a CPU
fallback and not a new communication kernel. HBM estimation is nonzero, but
bandwidth is an uncalibrated caller-provided HBM estimate and storage padding/
allocator peaks require separate capacity validation. Direct Topology
construction still emits the upstream TopologyFactory warning. This result
does not establish topology-factory, distributed, pooled, compiled, CRI or
full-model MLPerf support. No FBGEMM or torchlib-xpu source changes were needed.

The sections below describe the protected baseline before this experiment.

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