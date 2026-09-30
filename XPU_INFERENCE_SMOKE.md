# XPU inference packaging and recovery

## M1: local two-tile no-bag QUANT TW/RW/CW

Branch: `xpu/inference-r1-nobag-sharding`, based on `5795b987`. Scope is one
process on one PVC with two logical tiles (`xpu:0`, `xpu:1`),
`EmbeddingCollection` without pooling, INT4/INT8, DEVICE storage, bounds v1,
D=512 and row alignment 16. No process group, XCCL, pooled lookup, compile,
cache/UVM or pruning.

Runtime changes relative to `5795b987`:
- `create_infer_embedding_sharding` no longer rejects XPU RW/CW or
  `world_size>1`; other sharding types still raise.
- `QuantEmbeddingCollectionSharder` uses the base TW/RW/CW list for XPU.
- `SeqEmbeddingsAllToOne` drops the single-XPU identity shortcut and always
  calls `fbgemm::all_to_one_device`. This requires an `fbgemm_xpu` wheel that
  registers that operator for XPU (torchlib-xpu `c4c1c015` or later).

`QuantSequenceXpuTwoTileTest` plans, shards and runs TW, RW and CW with two
tables (64/48 rows, one shared by two features), and checks for each bitwidth:
- planner placement on both tiles (TW one table per tile, RW rows 32/32 and
  24/24, CW columns 256/256);
- parity with the CPU non-sharded reference for valid, empty-shard, all-empty
  and 12k-index inputs, output values on `xpu:0`, FP32, `[num_indices, 512]`;
- dispatch audit: every `fbgemm::` call has only XPU tensors, lookups and
  bounds run on both tiles, output routing uses `all_to_one_device` into
  `xpu:0`, RW bucketizes on `xpu:0`, and per-tile local indices match;
- bounds WARNING corrections (RW compared with CPU RW sharding, since RW
  applies bounds per shard) and FATAL errors;
- packed round trip: plain-tensor checkpoint of the sharded
  `0.embeddings.*` state (uint8 packed rows, FP16 scale/bias, shard metadata),
  loaded with `weights_only=True` into a corrupted fresh model; raw TBE
  buffers including padding match byte-for-byte, and load peak stays within
  twice the checkpoint tensor bytes per tile.

Observed contract details:
- TW `lengths()` of features served by `xpu:1` stay on `xpu:1`; values are
  routed to `xpu:0`. CUDA has the same behavior.
- RW output reconstruction performs two explicit device-to-host length reads
  per forward. The lookup operator keeps its known internal host reads.
- With registered TBEs (`register_tbes=True`), the sharded state dict also
  contains `*.tbes.N.*` runtime buffers whose names repeat across layouts.
  Load only `*.embeddings.*` keys from a checkpoint of a different layout:
  `strict=False` on a full non-sharded state dict silently overwrites
  `tbes.0.rows_per_table`/`D_offsets` of RW/CW shards.

Run on a host with both tiles visible:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCHREC_XPU_TESTS=1 OMP_NUM_THREADS=4
export TORCHREC_XPU_AUDIT_DIR=/new/audit-dir  # optional operator records
python -m pytest torchrec/distributed/tests/test_quant_sequence_xpu.py -v
```

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