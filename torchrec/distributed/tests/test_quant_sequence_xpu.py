import copy
import importlib
import io
import itertools
import json
import os
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode


@unittest.skipUnless(
    os.environ.get("TORCHREC_XPU_TESTS") == "1", "Set TORCHREC_XPU_TESTS=1"
)
class QuantSequenceXpuTest(unittest.TestCase):
    def test_planner_table_wise_round_trip(self):
        importlib.import_module("fbgemm_xpu")
        from fbgemm_gpu.split_embedding_configs import SparseType
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode
        from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
            IntNBitTableBatchedEmbeddingBagsCodegen,
        )
        from fbgemm_gpu.tbe.utils.quantize import quantize_embs
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel
        from torchrec.distributed.fused_params import FUSED_PARAM_REGISTER_TBE_BOOL
        from torchrec.distributed.global_settings import (
            get_propogate_device,
            set_propogate_device,
        )
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
        from torchrec.distributed.planner.shard_estimators import (
            EmbeddingPerfEstimator,
            EmbeddingStorageEstimator,
        )
        from torchrec.distributed.planner.types import ParameterConstraints
        from torchrec.distributed.quant_embedding import (
            QuantEmbeddingCollectionSharder,
            ShardedQuantEmbeddingCollection,
        )
        from torchrec.distributed.shard import _shard_modules
        from torchrec.distributed.types import ShardingEnv, ShardingType
        from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
        from torchrec.quant.embedding_modules import EmbeddingCollection
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        self.assertTrue(torch.xpu.is_available(), "A real XPU is required")
        self.assertFalse(torch.distributed.is_initialized())
        self.addCleanup(set_propogate_device, get_propogate_device())
        set_propogate_device(True)
        device = torch.device("xpu:0")

        class ObserveDispatch(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.calls = set()

            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                result = function(*args, **(kwargs or {}))
                if function._schema.name in (
                    "fbgemm::bounds_check_indices",
                    "fbgemm::int_nbit_split_embedding_codegen_lookup_function",
                ):
                    self.calls.add((function._schema.name, args[0].device.type))
                return result

        for bits, dimension in itertools.product((4, 8), (16, 512)):
            with self.subTest(bits=bits, dimension=dimension):
                tables = [
                    EmbeddingConfig(
                        name=f"table_{index}",
                        num_embeddings=rows,
                        embedding_dim=dimension,
                        data_type=DataType.INT4 if bits == 4 else DataType.INT8,
                        feature_names=[f"feature_{index}"],
                    )
                    for index, rows in enumerate((17, 23))
                ]
                weight_type = SparseType.INT4 if bits == 4 else SparseType.INT8
                packed_weights = {
                    table.name: quantize_embs(
                        torch.sin(
                            torch.arange(table.num_embeddings * dimension)
                            .float()
                            .reshape(-1, dimension)
                        ),
                        weight_type,
                    )
                    for table in tables
                }

                def make_model(target):
                    return torch.nn.Sequential(
                        EmbeddingCollection(
                            tables=copy.deepcopy(tables),
                            device=target,
                            output_dtype=torch.float32,
                            register_tbes=True,
                            row_alignment=16,
                            quant_state_dict_split_scale_bias=True,
                            table_name_to_quantized_weights=copy.deepcopy(
                                packed_weights
                            ),
                        )
                    ).eval()

                def enable_bounds(model):
                    modules = [
                        module
                        for module in model.modules()
                        if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen)
                    ]
                    self.assertTrue(modules)
                    for module in modules:
                        module.bounds_check_mode_int = BoundsCheckMode.WARNING.value
                        module.bounds_check_version = 1

                def features(target):
                    return KeyedJaggedTensor.from_lengths_sync(
                        keys=["feature_0", "feature_1"],
                        values=torch.tensor(
                            [0, 16, 3, 0, 22], dtype=torch.int32, device=target
                        ),
                        lengths=torch.tensor(
                            [2, 0, 1, 1, 1, 0], dtype=torch.int32, device=target
                        ),
                    )

                reference = make_model(torch.device("cpu"))
                unsharded = make_model(device)
                enable_bounds(reference)
                enable_bounds(unsharded)
                constraints = {
                    table.name: ParameterConstraints(
                        sharding_types=[ShardingType.TABLE_WISE.value],
                        compute_kernels=[EmbeddingComputeKernel.QUANT.value],
                    )
                    for table in tables
                }
                topology = Topology(world_size=1, compute_device="xpu", hbm_cap=2**30)
                sharder = QuantEmbeddingCollectionSharder(
                    fused_params={FUSED_PARAM_REGISTER_TBE_BOOL: True}
                )
                planner = EmbeddingShardingPlanner(
                    topology=topology,
                    batch_size=3,
                    constraints=constraints,
                    enumerator=EmbeddingEnumerator(
                        topology=topology,
                        batch_size=3,
                        constraints=constraints,
                        estimator=[
                            EmbeddingPerfEstimator(
                                topology=topology, is_inference=True
                            ),
                            EmbeddingStorageEstimator(topology=topology),
                        ],
                    ),
                )
                plan = planner.plan(unsharded, [sharder])
                print(f"INT{bits}, D={dimension} actual plan:", plan)
                self.assertEqual(set(plan.plan["0"]), {table.name for table in tables})
                selected = planner.get_selected_options()
                self.assertEqual(len(selected), len(tables))
                estimated_hbm = sum(
                    shard.storage.hbm for option in selected for shard in option.shards
                )
                self.assertGreaterEqual(
                    estimated_hbm,
                    sum(
                        table.num_embeddings * dimension * bits // 8 for table in tables
                    ),
                )
                self.assertLess(estimated_hbm, 2**30)
                print("Estimated HBM bytes:", estimated_hbm)
                for parameter in plan.plan["0"].values():
                    self.assertEqual(parameter.sharding_type, "table_wise")
                    self.assertEqual(parameter.compute_kernel, "quant")
                    self.assertEqual(parameter.ranks, [0])
                    for shard in parameter.sharding_spec.shards:
                        self.assertEqual(shard.placement.device(), device)

                def make_sharded():
                    model = _shard_modules(
                        module=make_model(device),
                        plan=plan,
                        sharders=[sharder],
                        device=device,
                        env=ShardingEnv.from_local(world_size=1, rank=0),
                    )
                    self.assertIsInstance(model[0], ShardedQuantEmbeddingCollection)
                    model.load_state_dict(unsharded.state_dict(), strict=True)
                    enable_bounds(model)
                    return model.eval()

                sharded = make_sharded()
                with torch.no_grad():
                    expected = reference(features(torch.device("cpu")))
                    ordinary = unsharded(features(device))
                    with ObserveDispatch() as observed:
                        actual = sharded(features(device))
                for operator in (
                    "bounds_check_indices",
                    "int_nbit_split_embedding_codegen_lookup_function",
                ):
                    self.assertIn((f"fbgemm::{operator}", "xpu"), observed.calls)

                def assert_outputs(outputs):
                    self.assertEqual(outputs.keys(), expected.keys())
                    for name in expected:
                        self.assertEqual(outputs[name].values().device.type, "xpu")
                        torch.testing.assert_close(
                            outputs[name].values().cpu(),
                            expected[name].values(),
                            rtol=1e-5,
                            atol=1e-6,
                        )
                        torch.testing.assert_close(
                            outputs[name].lengths().cpu(), expected[name].lengths()
                        )

                assert_outputs(ordinary)
                assert_outputs(actual)
                serialized = io.BytesIO()
                checkpoint = {
                    name: value.detach().clone()
                    for name, value in sharded.state_dict().items()
                }
                torch.save(checkpoint, serialized)
                serialized.seek(0)
                restored = make_sharded()
                with torch.no_grad():
                    for module in restored.modules():
                        if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen):
                            for weights, scale_bias in module.split_embedding_weights():
                                weights.zero_()
                                if scale_bias is not None:
                                    scale_bias.zero_()
                    cleared = restored(features(device))
                    self.assertTrue(
                        any(
                            not torch.allclose(
                                cleared[name].values().cpu(), expected[name].values()
                            )
                            for name in expected
                        )
                    )
                restored.load_state_dict(
                    torch.load(serialized, weights_only=False), strict=True
                )
                with torch.no_grad():
                    assert_outputs(restored(features(device)))
                self.assertFalse(torch.distributed.is_initialized())
                print(
                    f"INT{bits}, D={dimension}: planner, table-wise forward and checkpoint round-trip PASS"
                )

    def test_local_output_aliases_through_operator(self):
        importlib.import_module("fbgemm_xpu")
        from torchrec.distributed.dist_data import SeqEmbeddingsAllToOne

        self.assertTrue(torch.xpu.is_available())
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "fbgemm::all_to_one_device", "XPU"
            )
        )
        tensor = torch.arange(8, device="xpu:0")
        result = SeqEmbeddingsAllToOne(torch.device("xpu:0"), world_size=1)([tensor])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].device, tensor.device)
        self.assertEqual(result[0].data_ptr(), tensor.data_ptr())


TWO_TILE_ROWS = {"table_0": 64, "table_1": 48}
TWO_TILE_DIMENSION = 512
TWO_TILE_FEATURES = {
    "table_0": ["feature_0", "feature_2"],
    "table_1": ["feature_1"],
}
TWO_TILE_KEYS = ["feature_0", "feature_1", "feature_2"]
TWO_TILE_INPUTS = {
    "valid": [[0, 63], [], [31, 32, 17], [47], [0, 24], [23], [], [40], [5]],
    "one_row_shard_empty": [[3], [20], [], [], [0, 23], [], [7], [], []],
    "all_empty": [[] for _ in range(9)],
    "out_of_range": [[-2], [67], [], [], [51], [1], [2], [], []],
}


def _two_tile_input(name, device):
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    if name == "large":
        generator = torch.Generator().manual_seed(2026)
        segments = []
        for key in TWO_TILE_KEYS:
            table = next(t for t, f in TWO_TILE_FEATURES.items() if key in f)
            for length in (1500, 0, 2596):
                segments.append(
                    torch.randint(
                        0, TWO_TILE_ROWS[table], (length,), generator=generator
                    ).tolist()
                )
    else:
        segments = TWO_TILE_INPUTS[name]
    return KeyedJaggedTensor.from_lengths_sync(
        keys=TWO_TILE_KEYS,
        values=torch.tensor(
            [index for segment in segments for index in segment],
            dtype=torch.int32,
            device=device,
        ),
        lengths=torch.tensor(
            [len(segment) for segment in segments], dtype=torch.int32, device=device
        ),
    )


def _tensor_arguments(function, args, kwargs):
    names = [argument.name for argument in function._schema.arguments]
    bound = dict(zip(names, args))
    bound.update(kwargs)
    tensors = {}
    for name, value in bound.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        described = [
            (str(tensor.device), str(tensor.dtype), list(tensor.shape))
            for tensor in values
            if isinstance(tensor, torch.Tensor)
        ]
        if described:
            tensors[name] = described
    return bound, tensors


class AuditDispatch(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.lookup_indices = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        schema = function._schema
        bound, tensors = _tensor_arguments(function, args, kwargs)
        call = {
            "op": f"{schema.name}.{schema.overload_name or 'default'}",
            "tensors": tensors,
        }
        device_arguments = {
            name: str(value)
            for name, value in bound.items()
            if isinstance(value, torch.device)
        }
        if device_arguments:
            call["device_arguments"] = device_arguments
        if isinstance(result, torch.Tensor):
            call["result_device"] = str(result.device)
        self.calls.append(call)
        if schema.name == "fbgemm::int_nbit_split_embedding_codegen_lookup_function":
            self.lookup_indices.append(
                (bound["indices"].device, bound["indices"].cpu().tolist())
            )
        return result

    def summary(self):
        unique = {}
        for call in self.calls:
            key = repr(sorted(call.items()))
            unique.setdefault(key, dict(call, count=0))["count"] += 1
        return list(unique.values())

    def host_transfers(self):
        return sum(
            1
            for call in self.calls
            if any(device != "cpu" for device in _tensor_devices(call))
            and (
                call.get("result_device") == "cpu"
                or call["op"].startswith("aten::_local_scalar_dense")
            )
        )


def _tensor_devices(call):
    return {device for described in call["tensors"].values() for device, _, _ in described}


def logical_state_dict(model):
    # Registered TBEs add runtime buffers (D_offsets, rows_per_table, ...) whose
    # names repeat across differently sharded models while their values differ.
    return {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith("0.embeddings.")
    }


def _packed_host_copy(tensor):
    from torchrec.tensor_types import UIntXTensor

    wrapper = type(tensor).__name__ if isinstance(tensor, UIntXTensor) else None
    raw = tensor.view(torch.uint8) if wrapper else tensor
    return wrapper, raw.detach().cpu().clone()


def packed_checkpoint(state_dict):
    from torch.distributed._shard.sharded_tensor import ShardedTensorBase

    checkpoint = {}
    for name, value in state_dict.items():
        if isinstance(value, ShardedTensorBase):
            shards = []
            for shard in value.local_shards():
                wrapper, tensor = _packed_host_copy(shard.tensor)
                shards.append(
                    {
                        "offsets": list(shard.metadata.shard_offsets),
                        "sizes": list(shard.metadata.shard_sizes),
                        "placement": str(shard.metadata.placement),
                        "wrapper": wrapper,
                        "tensor": tensor,
                    }
                )
            checkpoint[name] = {"size": list(value.metadata().size), "shards": shards}
        elif isinstance(value, list):
            checkpoint[name] = {"list": [t.detach().cpu().clone() for t in value]}
        else:
            wrapper, tensor = _packed_host_copy(value)
            checkpoint[name] = {"wrapper": wrapper, "tensor": tensor}
    return checkpoint


def restore_packed_state_dict(checkpoint, destination):
    from torch.distributed._shard.sharded_tensor import (
        Shard,
        ShardedTensorBase,
        ShardedTensorMetadata,
        ShardMetadata,
    )
    from torchrec.tensor_types import UInt2Tensor, UInt4Tensor

    wrappers = {"UInt4Tensor": UInt4Tensor, "UInt2Tensor": UInt2Tensor}

    def wrap(entry, device):
        tensor = entry["tensor"].to(device)
        return wrappers[entry["wrapper"]](tensor) if entry["wrapper"] else tensor

    state = {}
    targets = destination.state_dict()
    for name, entry in checkpoint.items():
        target = targets[name]
        if isinstance(target, ShardedTensorBase):
            shards = []
            for saved in entry["shards"]:
                metadata = ShardMetadata(
                    shard_offsets=saved["offsets"],
                    shard_sizes=saved["sizes"],
                    placement=saved["placement"],
                )
                shards.append(
                    Shard(tensor=wrap(saved, metadata.placement.device()), metadata=metadata)
                )
            state[name] = ShardedTensorBase._init_from_local_shards_and_global_metadata(
                local_shards=shards,
                sharded_tensor_metadata=ShardedTensorMetadata(
                    shards_metadata=[shard.metadata for shard in shards],
                    size=torch.Size(entry["size"]),
                ),
            )
        elif isinstance(target, list):
            state[name] = [
                saved.to(tensor.device) for saved, tensor in zip(entry["list"], target)
            ]
        else:
            state[name] = wrap(entry, target.device)
    return state


def _checkpoint_tensors(entry):
    if "shards" in entry:
        return [shard["tensor"] for shard in entry["shards"]]
    if "list" in entry:
        return entry["list"]
    return [entry["tensor"]]


def _tensor_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@unittest.skipUnless(
    os.environ.get("TORCHREC_XPU_TESTS") == "1", "Set TORCHREC_XPU_TESTS=1"
)
class QuantSequenceXpuTwoTileTest(unittest.TestCase):
    def setUp(self):
        importlib.import_module("fbgemm_xpu")
        from torchrec.distributed.global_settings import (
            get_propogate_device,
            set_propogate_device,
        )

        self.assertTrue(torch.xpu.is_available(), "A real XPU is required")
        self.assertGreaterEqual(
            torch.xpu.device_count(), 2, "Two logical XPU tiles are required"
        )
        self.assertFalse(torch.distributed.is_initialized())
        self.addCleanup(set_propogate_device, get_propogate_device())
        set_propogate_device(True)
        self.audit = {}

    def tearDown(self):
        directory = os.environ.get("TORCHREC_XPU_AUDIT_DIR")
        if directory and self.audit:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, f"{self._testMethodName}.json"), "w") as output:
                json.dump(self.audit, output, indent=2, sort_keys=True)

    def _tables(self, bits):
        from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

        return [
            EmbeddingConfig(
                name=name,
                num_embeddings=rows,
                embedding_dim=TWO_TILE_DIMENSION,
                data_type=DataType.INT4 if bits == 4 else DataType.INT8,
                feature_names=list(TWO_TILE_FEATURES[name]),
            )
            for name, rows in TWO_TILE_ROWS.items()
        ]

    def _packed_weights(self, tables, bits):
        from fbgemm_gpu.split_embedding_configs import SparseType
        from fbgemm_gpu.tbe.utils.quantize import quantize_embs

        weight_type = SparseType.INT4 if bits == 4 else SparseType.INT8
        return {
            table.name: quantize_embs(
                torch.sin(
                    torch.arange(table.num_embeddings * TWO_TILE_DIMENSION)
                    .float()
                    .reshape(-1, TWO_TILE_DIMENSION)
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

        tbes = self._tbes(model)
        self.assertTrue(tbes)
        for tbe in tbes:
            tbe.bounds_check_mode_int = getattr(BoundsCheckMode, mode).value
            tbe.bounds_check_version = 1

    def _plan(self, model, tables, sharding_type, compute_device):
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
            table.name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
                min_partition=(
                    TWO_TILE_DIMENSION // 2
                    if sharding_type == ShardingType.COLUMN_WISE.value
                    else None
                ),
            )
            for table in tables
        }
        topology = Topology(
            world_size=2,
            local_world_size=2,
            compute_device=compute_device,
            hbm_cap=2**30,
        )
        sharder = QuantEmbeddingCollectionSharder(
            fused_params={FUSED_PARAM_REGISTER_TBE_BOOL: True}
        )
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=3,
            constraints=constraints,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=3,
                constraints=constraints,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )
        return planner.plan(model, [sharder]), sharder

    def _expected_shards(self, sharding_type, plan):
        from torchrec.distributed.types import ShardingType

        expected = {}
        for name, rows in TWO_TILE_ROWS.items():
            if sharding_type == ShardingType.TABLE_WISE.value:
                (rank,) = plan.plan["0"][name].ranks
                expected[name] = [([0, 0], [rows, TWO_TILE_DIMENSION], rank)]
            elif sharding_type == ShardingType.ROW_WISE.value:
                block = (rows + 1) // 2
                expected[name] = [
                    ([0, 0], [block, TWO_TILE_DIMENSION], 0),
                    ([block, 0], [rows - block, TWO_TILE_DIMENSION], 1),
                ]
            else:
                half = TWO_TILE_DIMENSION // 2
                expected[name] = [([0, 0], [rows, half], 0), ([0, half], [rows, half], 1)]
        return expected

    def _assert_plan(self, plan, sharding_type, expected):
        from torchrec.distributed.types import ShardingType

        for name, shards in expected.items():
            parameter = plan.plan["0"][name]
            self.assertEqual(parameter.sharding_type, sharding_type)
            self.assertEqual(parameter.compute_kernel, "quant")
            actual = [
                (
                    list(shard.shard_offsets),
                    list(shard.shard_sizes),
                    shard.placement.device(),
                )
                for shard in parameter.sharding_spec.shards
            ]
            self.assertEqual(
                sorted(actual, key=repr),
                sorted(
                    [
                        (offsets, sizes, torch.device(f"xpu:{rank}"))
                        for offsets, sizes, rank in shards
                    ],
                    key=repr,
                ),
            )
        if sharding_type == ShardingType.TABLE_WISE.value:
            self.assertEqual(
                {shards[0][2] for shards in expected.values()}, {0, 1}
            )

    def _shard(self, plan, sharder, tables, packed_weights, source, device):
        from torchrec.distributed.quant_embedding import ShardedQuantEmbeddingCollection
        from torchrec.distributed.shard import _shard_modules
        from torchrec.distributed.types import ShardingEnv

        model = _shard_modules(
            module=self._model(tables, packed_weights, device),
            plan=plan,
            sharders=[sharder],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )
        self.assertIsInstance(model[0], ShardedQuantEmbeddingCollection)
        self._load_logical(model, logical_state_dict(source))
        self._set_bounds(model, "WARNING")
        return model.eval()

    def _load_logical(self, model, state):
        result = model.load_state_dict(state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(
            [key for key in result.missing_keys if key.startswith("0.embeddings.")], []
        )

    def _assert_outputs(self, actual, expected, target):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for name, reference in expected.items():
            values = actual[name].values()
            self.assertEqual(values.device, target, name)
            self.assertEqual(values.dtype, torch.float32, name)
            self.assertEqual(values.shape, reference.values().shape, name)
            torch.testing.assert_close(
                values.cpu(), reference.values().cpu(), rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                actual[name].lengths().cpu(), reference.lengths().cpu()
            )

    def _expected_local_indices(self, sharding_type, plan, features):
        from torchrec.distributed.types import ShardingType

        values = features.to_dict()
        expected = {0: [], 1: []}
        for table, feature_names in TWO_TILE_FEATURES.items():
            rows = TWO_TILE_ROWS[table]
            for feature in feature_names:
                indices = values[feature].values().cpu().tolist()
                if sharding_type == ShardingType.TABLE_WISE.value:
                    (rank,) = plan.plan["0"][table].ranks
                    expected[rank] += indices
                elif sharding_type == ShardingType.COLUMN_WISE.value:
                    expected[0] += indices
                    expected[1] += indices
                else:
                    block = (rows + 1) // 2
                    for index in indices:
                        expected[index // block].append(index % block)
        return {rank: sorted(indices) for rank, indices in expected.items()}

    def _assert_audit(self, audit, sharding_type, plan, features):
        from torchrec.distributed.types import ShardingType

        for call in audit.calls:
            if call["op"].startswith("fbgemm::"):
                devices = _tensor_devices(call)
                self.assertTrue(
                    all(device.startswith("xpu") for device in devices),
                    f"{call['op']} has non-XPU tensors: {devices}",
                )
                if not call["op"].startswith("fbgemm::all_to_one_device"):
                    self.assertEqual(len(devices), 1, call)
        by_op = {}
        for call in audit.calls:
            by_op.setdefault(call["op"].split(".")[0], []).append(call)
        lookups = by_op.get("fbgemm::int_nbit_split_embedding_codegen_lookup_function", [])
        self.assertEqual(
            {device for call in lookups for device in _tensor_devices(call)},
            {"xpu:0", "xpu:1"},
        )
        self.assertEqual(
            {
                device
                for call in by_op.get("fbgemm::bounds_check_indices", [])
                for device in _tensor_devices(call)
            },
            {"xpu:0", "xpu:1"},
        )
        routing = by_op.get("fbgemm::all_to_one_device", [])
        self.assertTrue(routing, "output routing did not call all_to_one_device")
        for call in routing:
            self.assertEqual(call["device_arguments"]["target_device"], "xpu:0")
        self.assertIn(
            "xpu:1", {device for call in routing for device in _tensor_devices(call)}
        )
        if sharding_type == ShardingType.ROW_WISE.value:
            bucketize = by_op.get("fbgemm::block_bucketize_sparse_features_inference", [])
            self.assertTrue(bucketize)
            self.assertEqual(
                {device for call in bucketize for device in _tensor_devices(call)},
                {"xpu:0"},
            )
        actual = {0: [], 1: []}
        for device, indices in audit.lookup_indices:
            actual[device.index] += indices
        self.assertEqual(
            {rank: sorted(indices) for rank, indices in actual.items()},
            self._expected_local_indices(sharding_type, plan, features),
        )

    def _round_trip(
        self, sharded, make_sharded, device, bits, sharding_type, expected, audit
    ):
        from torchrec.distributed.quant_state import sharded_tbes_weights_spec
        from torchrec.distributed.types import ShardingType

        spec = sharded_tbes_weights_spec(sharded)
        self.assertTrue(spec)
        state = logical_state_dict(sharded)
        checkpoint = packed_checkpoint(state)
        scale_bias_copies = 2 if sharding_type == ShardingType.COLUMN_WISE.value else 1
        packed_bytes = {}
        for table, rows in TWO_TILE_ROWS.items():
            prefix = f"0.embeddings.{table}."
            weights = _checkpoint_tensors(checkpoint[prefix + "weight"])
            self.assertTrue(all(tensor.dtype == torch.uint8 for tensor in weights))
            self.assertEqual(
                _tensor_bytes(weights), rows * TWO_TILE_DIMENSION * bits // 8, table
            )
            packed_bytes[table] = _tensor_bytes(weights)
            for suffix in ("weight_qscale", "weight_qbias"):
                parameters = _checkpoint_tensors(checkpoint[prefix + suffix])
                self.assertTrue(
                    all(tensor.dtype == torch.float16 for tensor in parameters)
                )
                self.assertEqual(
                    sum(tensor.numel() for tensor in parameters),
                    scale_bias_copies * rows,
                    prefix + suffix,
                )
        serialized = io.BytesIO()
        torch.save(checkpoint, serialized)
        audit["checkpoint_bytes"] = serialized.tell()
        audit["packed_weight_bytes"] = packed_bytes
        tensor_bytes = sum(
            _tensor_bytes(_checkpoint_tensors(entry)) for entry in checkpoint.values()
        )
        audit["checkpoint_tensor_bytes"] = tensor_bytes
        audit["state_dict_types"] = {
            name: type(value).__name__ for name, value in state.items()
        }
        serialized.seek(0)
        source_buffers = [tbe.weights_dev.cpu().clone() for tbe in self._tbes(sharded)]

        restored = make_sharded()
        with torch.no_grad():
            for tbe in self._tbes(restored):
                for weights, scale, bias in tbe.split_embedding_weights_with_scale_bias(
                    split_scale_bias_mode=2
                ):
                    weights.fill_(0x5A)
                    scale.zero_()
                    bias.zero_()
            cleared = restored(_two_tile_input("valid", device))
        self.assertTrue(
            any(
                not torch.allclose(cleared[name].values().cpu(), reference.values())
                for name, reference in expected.items()
                if reference.values().numel()
            )
        )
        self.assertFalse(
            all(
                torch.equal(tbe.weights_dev.cpu(), buffer)
                for tbe, buffer in zip(self._tbes(restored), source_buffers)
            )
        )

        loaded = torch.load(serialized, weights_only=True)
        devices = range(torch.xpu.device_count())
        for index in devices:
            torch.xpu.synchronize(index)
            torch.xpu.reset_peak_memory_stats(index)
        allocated = [torch.xpu.memory_allocated(index) for index in devices]
        self._load_logical(restored, restore_packed_state_dict(loaded, restored))
        for index in devices:
            torch.xpu.synchronize(index)
        peaks = [
            torch.xpu.max_memory_allocated(index) - allocated[index] for index in devices
        ]
        audit["load_peak_device_bytes"] = peaks
        audit["reserved_after_load"] = [torch.xpu.memory_reserved(index) for index in devices]
        audit["free_after_load"] = [torch.xpu.mem_get_info(index)[0] for index in devices]
        for index in devices:
            self.assertLessEqual(peaks[index], 2 * tensor_bytes + 2**21)
        restored_tbes = self._tbes(restored)
        self.assertEqual(len(restored_tbes), len(source_buffers))
        for tbe, buffer in zip(restored_tbes, source_buffers):
            self.assertTrue(torch.equal(tbe.weights_dev.cpu(), buffer))
        self.assertEqual(sharded_tbes_weights_spec(restored), spec)
        return restored

    def _run_two_tile(self, sharding_type):
        from torchrec.distributed.types import ShardingType

        target = torch.device("xpu:0")
        cpu = torch.device("cpu")
        for bits in (4, 8):
            with self.subTest(bits=bits, sharding_type=sharding_type):
                record = self.audit.setdefault(f"int{bits}", {})
                tables = self._tables(bits)
                packed_weights = self._packed_weights(tables, bits)
                reference = self._model(tables, packed_weights, cpu)
                unsharded = self._model(tables, packed_weights, target)
                self._set_bounds(reference, "WARNING")
                self._set_bounds(unsharded, "WARNING")
                plan, sharder = self._plan(unsharded, tables, sharding_type, "xpu")
                expected_shards = self._expected_shards(sharding_type, plan)
                self._assert_plan(plan, sharding_type, expected_shards)
                record["plan"] = str(plan)

                def make_sharded():
                    return self._shard(
                        plan, sharder, tables, packed_weights, unsharded, target
                    )

                sharded = make_sharded()
                self.assertEqual(
                    {tbe.weights_dev.device.index for tbe in self._tbes(sharded)}, {0, 1}
                )
                expected = {}
                for input_name in ("valid", "one_row_shard_empty", "all_empty", "large"):
                    with torch.no_grad():
                        expected[input_name] = reference(_two_tile_input(input_name, cpu))
                        self._assert_outputs(
                            unsharded(_two_tile_input(input_name, target)),
                            expected[input_name],
                            target,
                        )
                        features = _two_tile_input(input_name, target)
                        with AuditDispatch() as audit:
                            actual = sharded(features)
                        self._assert_outputs(actual, expected[input_name], target)
                        self._assert_outputs(sharded(features), expected[input_name], target)
                    self._assert_audit(audit, sharding_type, plan, features)
                    record[input_name] = {
                        "operators": audit.summary(),
                        "device_to_host_copies": audit.host_transfers(),
                        "lengths_devices": {
                            name: str(jagged.lengths().device)
                            for name, jagged in actual.items()
                        },
                    }

                restored = self._round_trip(
                    sharded,
                    make_sharded,
                    target,
                    bits,
                    sharding_type,
                    expected["valid"],
                    record,
                )
                with torch.no_grad():
                    for input_name in ("valid", "one_row_shard_empty", "all_empty", "large"):
                        self._assert_outputs(
                            restored(_two_tile_input(input_name, target)),
                            expected[input_name],
                            target,
                        )

                if sharding_type == ShardingType.ROW_WISE.value:
                    bounds_reference = self._shard_cpu_row_wise(
                        tables, packed_weights, reference
                    )
                else:
                    bounds_reference = reference
                warnings = sum(int(tbe.bounds_check_warning.item()) for tbe in self._tbes(restored))
                with torch.no_grad():
                    self._assert_outputs(
                        restored(_two_tile_input("out_of_range", target)),
                        bounds_reference(_two_tile_input("out_of_range", cpu)),
                        target,
                    )
                corrected = (
                    sum(int(tbe.bounds_check_warning.item()) for tbe in self._tbes(restored))
                    - warnings
                )
                self.assertGreater(corrected, 0)
                record["bounds_warning_corrections"] = corrected
                self._set_bounds(restored, "FATAL")
                with torch.no_grad(), self.assertRaises(RuntimeError):
                    restored(_two_tile_input("out_of_range", target))
                    for index in range(2):
                        torch.xpu.synchronize(index)
                self._set_bounds(restored, "WARNING")
                with torch.no_grad():
                    self._assert_outputs(
                        restored(_two_tile_input("valid", target)), expected["valid"], target
                    )
                self.assertFalse(torch.distributed.is_initialized())

    def _shard_cpu_row_wise(self, tables, packed_weights, reference):
        from torchrec.distributed.quant_embedding import ShardedQuantEmbeddingCollection
        from torchrec.distributed.shard import _shard_modules
        from torchrec.distributed.types import ShardingEnv, ShardingType

        cpu = torch.device("cpu")
        model = self._model(tables, packed_weights, cpu)
        plan, sharder = self._plan(model, tables, ShardingType.ROW_WISE.value, "cpu")
        sharded = _shard_modules(
            module=model,
            plan=plan,
            sharders=[sharder],
            device=cpu,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )
        self.assertIsInstance(sharded[0], ShardedQuantEmbeddingCollection)
        self._load_logical(sharded, logical_state_dict(reference))
        self._set_bounds(sharded, "WARNING")
        return sharded.eval()

    def test_output_routing_to_each_tile(self):
        from torchrec.distributed.dist_data import SeqEmbeddingsAllToOne

        generator = torch.Generator().manual_seed(7)
        expected = [
            torch.randn((5, TWO_TILE_DIMENSION), generator=generator),
            torch.randn((3, TWO_TILE_DIMENSION), generator=generator),
        ]
        for target_index in range(2):
            with self.subTest(target=target_index):
                target = torch.device(f"xpu:{target_index}")
                streams = [torch.xpu.Stream(device=f"xpu:{index}") for index in range(2)]
                inputs = []
                for index, reference in enumerate(expected):
                    with torch.xpu.stream(streams[index]):
                        inputs.append(reference.to(f"xpu:{index}") * 2)
                for index in range(2):
                    torch.xpu.current_stream(index).wait_stream(streams[index])
                outputs = SeqEmbeddingsAllToOne(target, world_size=2)(inputs)
                self.assertEqual(len(outputs), 2)
                self.assertEqual(
                    outputs[target_index].data_ptr(), inputs[target_index].data_ptr()
                )
                del inputs
                results = [output.cpu() for output in outputs]
                for output, result, reference in zip(outputs, results, expected):
                    self.assertEqual(output.device, target)
                    torch.testing.assert_close(result, reference * 2)

    def test_table_wise_two_tiles(self):
        self._run_two_tile("table_wise")

    def test_row_wise_two_tiles(self):
        self._run_two_tile("row_wise")

    def test_column_wise_two_tiles(self):
        self._run_two_tile("column_wise")


class QuantXpuPlannerContractTest(unittest.TestCase):
    def test_topology_bandwidth_and_placements(self):
        from torchrec.distributed.planner.constants import kernel_bw_lookup
        from torchrec.distributed.planner.types import Topology
        from torchrec.distributed.planner.utils import placement as planner_placement
        from torchrec.distributed.sharding_plan import placement

        topology = Topology(world_size=1, compute_device="xpu", hbm_cap=4096)
        self.assertEqual(topology.devices[0].storage.hbm, 4096)
        self.assertEqual(kernel_bw_lookup("xpu", "quant", 123.0, 45.0, 6.0), 123.0)
        self.assertIsNone(kernel_bw_lookup("xpu", "quant_uvm", 123.0, 45.0, 6.0))
        self.assertEqual(placement("xpu", 0, 1), "rank:0/xpu:0")
        self.assertEqual(planner_placement("xpu", 0, 1), "rank:0/xpu:0")
        self.assertEqual(placement("cuda", 1, 2), "rank:1/cuda:1")
        self.assertEqual(placement("cpu", 0, 1), "rank:0/cpu")

    def test_validated_sharding_types(self):
        from unittest.mock import patch
        from torchrec.distributed.quant_embedding import (
            QuantEmbeddingCollectionSharder,
            create_infer_embedding_sharding,
        )
        from torchrec.distributed.types import ShardingEnv

        sharder = QuantEmbeddingCollectionSharder()
        self.assertEqual(
            sharder.sharding_types("xpu"), ["table_wise", "row_wise", "column_wise"]
        )
        with patch(
            "torchrec.distributed.quant_embedding.get_device_from_sharding_infos",
            return_value="xpu",
        ):
            for sharding_type in ("table_row_wise", "table_column_wise", "data_parallel"):
                with self.subTest(sharding_type=sharding_type):
                    with self.assertRaisesRegex(ValueError, "Sharding type not supported"):
                        create_infer_embedding_sharding(
                            sharding_type,
                            [],
                            ShardingEnv.from_local(2, 0),
                            torch.device("xpu:0"),
                        )
