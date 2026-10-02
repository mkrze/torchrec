import copy
import importlib
import json
import math
import os
import time
import unittest

import torch

from torchrec.distributed.tests.test_quant_sequence_xpu import (
    _tensor_devices,
    AuditDispatch,
    logical_state_dict,
)

DIMENSION = 512
# Shapes follow DLRM-v3: one large row-wise table shared by history and
# candidate features, small table-wise user and category tables.
TABLES = {
    "item_id": (1003, ["item_id", "item_candidate_id"]),
    "user_id": (97, ["user_id"]),
    "item_category_id": (128, ["item_category_id", "item_candidate_category_id"]),
}
KEYS = [feature for _, features in TABLES.values() for feature in features]
BATCH = 3


def _world_size():
    return int(os.environ.get("TORCHREC_XPU_WORLD_SIZE", torch.xpu.device_count()))


def _features(name, device, world_size):
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    generator = torch.Generator().manual_seed(2026)
    segments = []
    for key in KEYS:
        rows = next(rows for rows, features in TABLES.values() if key in features)
        for sample in range(BATCH):
            if name == "all_empty" or (name == "mixed" and sample == 1):
                segments.append([])
            elif name == "boundaries":
                block = math.ceil(rows / world_size)
                edges = {0, rows - 1}
                for start in range(0, rows, block):
                    edges.update((start, min(rows, start + block) - 1))
                segments.append(sorted(edges))
            else:
                length = 16384 if name == "long" and key == "item_id" else 7
                segments.append(
                    torch.randint(0, rows, (length,), generator=generator).tolist()
                )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=KEYS,
        values=torch.tensor(
            [index for segment in segments for index in segment],
            dtype=torch.int32,
            device=device,
        ),
        lengths=torch.tensor(
            [len(segment) for segment in segments], dtype=torch.int32, device=device
        ),
    )


@unittest.skipUnless(
    os.environ.get("TORCHREC_XPU_TESTS") == "1", "Set TORCHREC_XPU_TESTS=1"
)
class QuantSequenceXpuMultiDeviceTest(unittest.TestCase):
    """Row-wise item table over every visible XPU, one process, no collectives."""

    def setUp(self):
        importlib.import_module("fbgemm_xpu")
        from torchrec.distributed.global_settings import (
            get_propogate_device,
            set_propogate_device,
        )

        self.assertTrue(torch.xpu.is_available(), "A real XPU is required")
        self.world_size = _world_size()
        self.assertGreaterEqual(self.world_size, 2)
        self.assertLessEqual(self.world_size, torch.xpu.device_count())
        self.assertFalse(torch.distributed.is_initialized())
        self.addCleanup(set_propogate_device, get_propogate_device())
        set_propogate_device(True)
        self.record = {"world_size": self.world_size}

    def tearDown(self):
        directory = os.environ.get("TORCHREC_XPU_AUDIT_DIR")
        if directory:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{self._testMethodName}-ws{self.world_size}.json")
            with open(path, "w") as output:
                json.dump(self.record, output, indent=2, sort_keys=True)

    def _tables(self, bits):
        from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

        return [
            EmbeddingConfig(
                name=name,
                num_embeddings=rows,
                embedding_dim=DIMENSION,
                data_type=DataType.INT4 if bits == 4 else DataType.INT8,
                feature_names=list(features),
            )
            for name, (rows, features) in TABLES.items()
        ]

    def _packed_weights(self, tables, bits):
        from fbgemm_gpu.split_embedding_configs import SparseType
        from fbgemm_gpu.tbe.utils.quantize import quantize_embs

        weight_type = SparseType.INT4 if bits == 4 else SparseType.INT8
        return {
            table.name: quantize_embs(
                torch.sin(
                    torch.arange(table.num_embeddings * DIMENSION)
                    .float()
                    .reshape(-1, DIMENSION)
                    * 0.01
                    + index
                ),
                weight_type,
            )
            for index, table in enumerate(tables)
        }

    def _model(self, tables, packed_weights, device):
        from torchrec.quant.embedding_modules import EmbeddingCollection

        return torch.nn.Sequential(
            EmbeddingCollection(
                tables=copy.deepcopy(tables),
                device=device,
                output_dtype=torch.float32,
                register_tbes=True,
                row_alignment=16,
                quant_state_dict_split_scale_bias=True,
                table_name_to_quantized_weights=copy.deepcopy(packed_weights),
            )
        ).eval()

    def _tbes(self, model):
        from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
            IntNBitTableBatchedEmbeddingBagsCodegen,
        )

        return [
            module
            for module in model.modules()
            if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen)
        ]

    def _set_bounds(self, model, mode):
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode

        for tbe in self._tbes(model):
            tbe.bounds_check_mode_int = getattr(BoundsCheckMode, mode).value
            tbe.bounds_check_version = 1

    def _plan(self, model, sharding_types):
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel
        from torchrec.distributed.fused_params import FUSED_PARAM_REGISTER_TBE_BOOL
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
        from torchrec.distributed.planner.shard_estimators import (
            EmbeddingPerfEstimator,
            EmbeddingStorageEstimator,
        )
        from torchrec.distributed.planner.types import ParameterConstraints
        from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
        from torchrec.distributed.types import ShardingType

        constraints = {
            name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
                min_partition=(
                    DIMENSION // self.world_size
                    if sharding_type == ShardingType.COLUMN_WISE.value
                    else None
                ),
            )
            for name, sharding_type in sharding_types.items()
        }
        topology = Topology(
            world_size=self.world_size,
            local_world_size=self.world_size,
            compute_device="xpu",
            hbm_cap=2**30,
        )
        sharder = QuantEmbeddingCollectionSharder(
            fused_params={FUSED_PARAM_REGISTER_TBE_BOOL: True}
        )
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=BATCH,
            constraints=constraints,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=BATCH,
                constraints=constraints,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )
        return planner.plan(model, [sharder]), sharder

    def _shard(self, plan, sharder, tables, packed_weights, source, device):
        from torchrec.distributed.quant_embedding import ShardedQuantEmbeddingCollection
        from torchrec.distributed.shard import _shard_modules
        from torchrec.distributed.types import ShardingEnv

        model = _shard_modules(
            module=self._model(tables, packed_weights, device),
            plan=plan,
            sharders=[sharder],
            device=device,
            env=ShardingEnv.from_local(world_size=self.world_size, rank=0),
        )
        self.assertIsInstance(model[0], ShardedQuantEmbeddingCollection)
        result = model.load_state_dict(logical_state_dict(source), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(
            [key for key in result.missing_keys if key.startswith("0.embeddings.")], []
        )
        self._set_bounds(model, "WARNING")
        return model.eval()

    def _assert_outputs(self, actual, expected, target):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for name, reference in expected.items():
            values = actual[name].values()
            self.assertEqual(values.device, target, name)
            self.assertEqual(values.shape, reference.values().shape, name)
            torch.testing.assert_close(
                values.cpu(), reference.values().cpu(), rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                actual[name].lengths().cpu(), reference.lengths().cpu()
            )

    def _shard_placements(self, plan):
        return {
            name: [
                (
                    list(shard.shard_offsets),
                    list(shard.shard_sizes),
                    str(shard.placement.device()),
                )
                for shard in parameter.sharding_spec.shards
            ]
            for name, parameter in plan.plan["0"].items()
        }

    def _run(self, sharding_types, target_index, inputs):
        from torchrec.distributed.types import ShardingType

        target = torch.device(f"xpu:{target_index}")
        cpu = torch.device("cpu")
        all_devices = {f"xpu:{index}" for index in range(self.world_size)}
        for bits in (4, 8):
            with self.subTest(bits=bits, target=str(target)):
                record = self.record.setdefault(f"int{bits}-target{target_index}", {})
                tables = self._tables(bits)
                packed_weights = self._packed_weights(tables, bits)
                reference = self._model(tables, packed_weights, cpu)
                unsharded = self._model(tables, packed_weights, target)
                self._set_bounds(reference, "WARNING")
                self._set_bounds(unsharded, "WARNING")
                plan, sharder = self._plan(unsharded, sharding_types)
                placements = self._shard_placements(plan)
                record["placements"] = placements
                item_devices = {device for _, _, device in placements["item_id"]}
                self.assertEqual(item_devices, all_devices)
                if sharding_types["item_id"] == ShardingType.ROW_WISE.value:
                    block = math.ceil(TABLES["item_id"][0] / self.world_size)
                    self.assertTrue(
                        all(sizes[0] <= block for _, sizes, _ in placements["item_id"])
                    )
                started = time.perf_counter()
                sharded = self._shard(
                    plan, sharder, tables, packed_weights, unsharded, target
                )
                record["shard_seconds"] = time.perf_counter() - started
                self.assertEqual(
                    {f"xpu:{tbe.weights_dev.device.index}" for tbe in self._tbes(sharded)},
                    all_devices,
                )
                for input_name in inputs:
                    features = _features(input_name, target, self.world_size)
                    with torch.no_grad():
                        expected = reference(_features(input_name, cpu, self.world_size))
                        with AuditDispatch() as audit:
                            actual = sharded(features)
                        for index in range(self.world_size):
                            torch.xpu.synchronize(index)
                        self._assert_outputs(actual, expected, target)
                        started = time.perf_counter()
                        repeated = sharded(features)
                        for index in range(self.world_size):
                            torch.xpu.synchronize(index)
                        elapsed = time.perf_counter() - started
                        self._assert_outputs(repeated, expected, target)
                    by_op = {}
                    for call in audit.calls:
                        if call["op"].startswith("fbgemm::"):
                            self.assertTrue(
                                all(device.startswith("xpu") for device in _tensor_devices(call)),
                                call,
                            )
                        by_op.setdefault(call["op"].split(".")[0], []).append(call)
                    lookups = by_op.get(
                        "fbgemm::int_nbit_split_embedding_codegen_lookup_function", []
                    )
                    lookup_devices = {
                        device for call in lookups for device in _tensor_devices(call)
                    }
                    routing = by_op.get("fbgemm::all_to_one_device", [])
                    if input_name != "all_empty":
                        self.assertEqual(lookup_devices, all_devices, input_name)
                        self.assertTrue(routing, "output routing did not call all_to_one_device")
                    for call in routing:
                        self.assertEqual(call["device_arguments"]["target_device"], str(target))
                    record[input_name] = {
                        "lookup_devices": sorted(lookup_devices),
                        "routing_calls": len(routing),
                        "device_to_host_copies": audit.host_transfers(),
                        "indices": int(features.values().numel()),
                        "second_forward_seconds": elapsed,
                    }

    def test_row_wise_item_table_wise_small_tables(self):
        from torchrec.distributed.types import ShardingType

        sharding_types = {
            "item_id": ShardingType.ROW_WISE.value,
            "user_id": ShardingType.TABLE_WISE.value,
            "item_category_id": ShardingType.TABLE_WISE.value,
        }
        for target_index in (0, self.world_size - 1):
            self._run(sharding_types, target_index, ("boundaries", "mixed", "all_empty", "long"))

    def test_column_wise_item(self):
        from torchrec.distributed.types import ShardingType

        if DIMENSION % (4 * self.world_size):
            self.skipTest("column shards must keep D divisible by four")
        sharding_types = {
            "item_id": ShardingType.COLUMN_WISE.value,
            "user_id": ShardingType.TABLE_WISE.value,
            "item_category_id": ShardingType.TABLE_WISE.value,
        }
        self._run(sharding_types, 0, ("boundaries", "mixed", "long"))
