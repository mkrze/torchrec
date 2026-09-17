import copy
import importlib
import io
import itertools
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

    def test_local_output_identity(self):
        importlib.import_module("fbgemm_xpu")
        from torchrec.distributed.dist_data import SeqEmbeddingsAllToOne

        self.assertTrue(torch.xpu.is_available())
        tensor = torch.arange(8, device="xpu:0")
        tensors = [tensor]
        result = SeqEmbeddingsAllToOne(torch.device("xpu:0"), world_size=1)(tensors)
        self.assertIs(result, tensors)
        self.assertIs(result[0], tensor)


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

    def test_unvalidated_sharding_is_not_enabled(self):
        from unittest.mock import patch
        from torchrec.distributed.quant_embedding import (
            QuantEmbeddingCollectionSharder,
            create_infer_embedding_sharding,
        )
        from torchrec.distributed.types import ShardingEnv

        sharder = QuantEmbeddingCollectionSharder()
        self.assertEqual(sharder.sharding_types("xpu"), ["table_wise"])
        self.assertIn("row_wise", sharder.sharding_types("cuda"))
        with patch(
            "torchrec.distributed.quant_embedding.get_device_from_sharding_infos",
            return_value="xpu",
        ):
            for sharding_type, world_size in (
                ("row_wise", 1),
                ("column_wise", 1),
                ("table_wise", 2),
            ):
                with self.subTest(sharding_type=sharding_type, world_size=world_size):
                    with self.assertRaisesRegex(
                        ValueError, "requires table_wise and world_size=1"
                    ):
                        create_infer_embedding_sharding(
                            sharding_type,
                            [],
                            ShardingEnv.from_local(world_size, 0),
                            torch.device("xpu:0"),
                        )
