#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Sharded QuantEmbeddingCollection (sequence, no-bag) on Intel XPU devices.

The sharding tests run on XPU with an XPU build of the FBGEMM operators (for
example the fbgemm-xpu plugin from intel/torchlib-xpu) and are skipped
elsewhere; they shard over the visible devices of one process, with the default
set_propogate_device(False), and compare with the unsharded module on CPU. The
planner tests need no device.
"""

import copy
import importlib
import importlib.util
import math
import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils.quantize import quantize_embs
from torchrec.distributed.dist_data import SeqEmbeddingsAllToOne
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fused_params import FUSED_PARAM_REGISTER_TBE_BOOL
from torchrec.distributed.global_settings import (
    get_propogate_device,
    set_propogate_device,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import kernel_bw_lookup
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.planner.utils import placement as planner_placement
from torchrec.distributed.quant_embedding import (
    create_infer_embedding_sharding,
    QuantEmbeddingCollectionSharder,
    ShardedQuantEmbeddingCollection,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import placement
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

# Rows of 512 INT4 or INT8 values plus a 4-byte scale and bias are 260 and 516
# bytes, so the tables need padding to 16-byte rows.
DIMENSION = 512
# A large table shared by two features and two small tables, as in DLRM-v3.
TABLES: Dict[str, Tuple[int, List[str]]] = {
    "item_id": (1003, ["item_id", "item_candidate_id"]),
    "user_id": (97, ["user_id"]),
    "item_category_id": (128, ["item_category_id", "item_candidate_category_id"]),
}
KEYS: List[str] = [feature for _, features in TABLES.values() for feature in features]
BATCH = 3
INPUTS = ("boundaries", "mixed", "all_empty", "long")


def _xpu_device_count() -> int:
    if not torch.xpu.is_available() or importlib.util.find_spec("fbgemm_xpu") is None:
        return 0
    importlib.import_module("fbgemm_xpu")
    return torch.xpu.device_count()


XPU_DEVICES: int = _xpu_device_count()


def _tables(bits: int) -> List[EmbeddingConfig]:
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


def _packed_weights(
    tables: List[EmbeddingConfig], bits: int
) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    weight_type = SparseType.INT4 if bits == 4 else SparseType.INT8
    return {
        table.name: quantize_embs(
            torch.sin(
                torch.arange(table.num_embeddings * DIMENSION)
                .float()
                .view(-1, DIMENSION)
                * 0.01
                + index
            ),
            weight_type,
        )
        for index, table in enumerate(tables)
    }


def _model(
    tables: List[EmbeddingConfig],
    weights: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    device: torch.device,
) -> torch.nn.Module:
    model = torch.nn.Sequential(
        QuantEmbeddingCollection(
            tables=copy.deepcopy(tables),
            device=device,
            output_dtype=torch.float32,
            register_tbes=True,
            row_alignment=16,
            quant_state_dict_split_scale_bias=True,
            # pyrefly: ignore[bad-argument-type]
            table_name_to_quantized_weights=copy.deepcopy(weights),
        )
    ).eval()
    _warn_on_out_of_bounds(model)
    return model


def _warn_on_out_of_bounds(model: torch.nn.Module) -> None:
    # fbgemm-xpu implements version 1 of the indices bounds check.
    for module in model.modules():
        if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen):
            module.bounds_check_mode_int = BoundsCheckMode.WARNING.value
            module.bounds_check_version = 1


def _features(kind: str, device: torch.device, world_size: int) -> KeyedJaggedTensor:
    generator = torch.Generator().manual_seed(2026)
    segments: List[List[int]] = []
    for key in KEYS:
        rows = next(rows for rows, features in TABLES.values() if key in features)
        for sample in range(BATCH):
            if kind == "all_empty" or (kind == "mixed" and sample == 1):
                segments.append([])
            elif kind == "boundaries":
                # First and last row of every row-wise block.
                block = math.ceil(rows / world_size)
                edges = {0, rows - 1}
                for start in range(0, rows, block):
                    edges.update((start, min(rows, start + block) - 1))
                segments.append(sorted(edges))
            else:
                length = 16384 if kind == "long" and key == "item_id" else 7
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


class QuantXpuPlannerTest(unittest.TestCase):
    def test_topology_bandwidth_and_placements(self) -> None:
        topology = Topology(world_size=1, compute_device="xpu", hbm_cap=4096)
        self.assertEqual(topology.devices[0].storage.hbm, 4096)
        self.assertEqual(kernel_bw_lookup("xpu", "quant", 123.0, 45.0, 6.0), 123.0)
        self.assertIsNone(kernel_bw_lookup("xpu", "quant_uvm", 123.0, 45.0, 6.0))
        self.assertEqual(placement("xpu", 1, 2), "rank:1/xpu:1")
        self.assertEqual(planner_placement("xpu", 1, 2), "rank:1/xpu:1")

    def test_sharding_types(self) -> None:
        sharder = QuantEmbeddingCollectionSharder()
        self.assertEqual(
            sharder.sharding_types("xpu"), ["table_wise", "row_wise", "column_wise"]
        )
        with patch(
            "torchrec.distributed.quant_embedding.get_device_from_sharding_infos",
            return_value="xpu",
        ):
            for sharding_type in (
                "table_row_wise",
                "table_column_wise",
                "data_parallel",
            ):
                with self.subTest(sharding_type=sharding_type):
                    with self.assertRaisesRegex(
                        ValueError, "Sharding type not supported"
                    ):
                        create_infer_embedding_sharding(
                            sharding_type,
                            [],
                            ShardingEnv.from_local(2, 0),
                            torch.device("xpu:0"),
                        )


@unittest.skipIf(XPU_DEVICES < 1, "requires an XPU device and fbgemm-xpu")
class QuantSequenceXpuShardingTest(unittest.TestCase):
    def setUp(self) -> None:
        # The XPU device must reach every lookup without the global flag.
        self.addCleanup(set_propogate_device, get_propogate_device())
        set_propogate_device(False)

    def _plan_and_shard(
        self,
        tables: List[EmbeddingConfig],
        weights: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        world_size: int,
        sharding_types: Dict[str, str],
        target: torch.device,
    ) -> torch.nn.Module:
        constraints = {
            name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
                min_partition=(
                    DIMENSION // world_size
                    if sharding_type == ShardingType.COLUMN_WISE.value
                    else None
                ),
            )
            for name, sharding_type in sharding_types.items()
        }
        topology = Topology(
            world_size=world_size,
            local_world_size=world_size,
            compute_device="xpu",
            hbm_cap=2**30,
        )
        sharder = QuantEmbeddingCollectionSharder(
            fused_params={FUSED_PARAM_REGISTER_TBE_BOOL: True}
        )
        unsharded = _model(tables, weights, target)
        plan = EmbeddingShardingPlanner(
            topology=topology, batch_size=BATCH, constraints=constraints
        ).plan(unsharded, [sharder])
        devices = {torch.device(f"xpu:{index}") for index in range(world_size)}
        # pyrefly: ignore[missing-attribute]
        for name, parameter in plan.plan["0"].items():
            shards = parameter.sharding_spec.shards
            self.assertTrue({s.placement.device() for s in shards} <= devices, name)
            if sharding_types[name] != ShardingType.TABLE_WISE.value:
                self.assertEqual(len(shards), world_size, name)
        sharded = _shard_modules(
            module=_model(tables, weights, target),
            plan=plan,
            sharders=[sharder],
            device=target,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
        )
        # pyrefly: ignore[bad-index]
        self.assertIsInstance(sharded[0], ShardedQuantEmbeddingCollection)
        self._assert_tbes_on_xpu(sharded, world_size)
        # Sharding builds the TBEs empty; load the quantized rows from the
        # unsharded module (TBE runtime buffers are not part of the tables).
        result = sharded.load_state_dict(
            {
                name: value
                for name, value in unsharded.state_dict().items()
                if name.startswith("0.embeddings.")
            },
            strict=False,
        )
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(
            [key for key in result.missing_keys if key.startswith("0.embeddings.")], []
        )
        _warn_on_out_of_bounds(sharded)
        return sharded.eval()

    def _assert_tbes_on_xpu(self, sharded: torch.nn.Module, world_size: int) -> None:
        tbes = [
            module
            for module in sharded.modules()
            if isinstance(module, IntNBitTableBatchedEmbeddingBagsCodegen)
        ]
        self.assertTrue(tbes)
        devices = {torch.device(f"xpu:{index}") for index in range(world_size)}
        for tbe in tbes:
            self.assertIn(tbe.weights_dev.device, devices)
            for name, buffer in tbe.named_buffers():
                if buffer.numel():
                    self.assertEqual(buffer.device.type, "xpu", name)
        self.assertFalse(torch.cuda.is_initialized())

    def _assert_outputs(
        self,
        actual: Dict[str, JaggedTensor],
        expected: Dict[str, JaggedTensor],
        target: torch.device,
    ) -> None:
        self.assertEqual(set(actual), set(expected))
        for name, reference in expected.items():
            self.assertEqual(actual[name].values().device, target, name)
            torch.testing.assert_close(
                actual[name].values().cpu(), reference.values(), rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                actual[name].lengths().cpu(), reference.lengths().cpu()
            )

    def _check(
        self,
        world_size: int,
        sharding_types: Dict[str, str],
        targets: Tuple[int, ...],
        inputs: Tuple[str, ...] = INPUTS,
    ) -> None:
        cpu = torch.device("cpu")
        for bits in (4, 8):
            tables = _tables(bits)
            weights = _packed_weights(tables, bits)
            reference = _model(tables, weights, cpu)
            for target_index in targets:
                target = torch.device(f"xpu:{target_index}")
                sharded = self._plan_and_shard(
                    tables, weights, world_size, sharding_types, target
                )
                for kind in inputs:
                    with self.subTest(bits=bits, target=target_index, input=kind):
                        with torch.no_grad():
                            expected = reference(_features(kind, cpu, world_size))
                            actual = sharded(_features(kind, target, world_size))
                        self._assert_outputs(actual, expected, target)

    def test_table_wise(self) -> None:
        for world_size in (1, 2) if XPU_DEVICES >= 2 else (1,):
            with self.subTest(world_size=world_size):
                self._check(
                    world_size,
                    {name: ShardingType.TABLE_WISE.value for name in TABLES},
                    tuple(range(world_size)),
                )

    @unittest.skipIf(XPU_DEVICES < 2, "requires two XPU devices")
    def test_row_wise(self) -> None:
        self._check(2, {name: ShardingType.ROW_WISE.value for name in TABLES}, (0, 1))

    @unittest.skipIf(XPU_DEVICES < 2, "requires two XPU devices")
    def test_column_wise(self) -> None:
        self._check(
            2, {name: ShardingType.COLUMN_WISE.value for name in TABLES}, (0, 1)
        )

    @unittest.skipIf(XPU_DEVICES < 2, "requires two XPU devices")
    def test_row_wise_large_table_over_all_devices(self) -> None:
        sharding_types = {
            "item_id": ShardingType.ROW_WISE.value,
            "user_id": ShardingType.TABLE_WISE.value,
            "item_category_id": ShardingType.TABLE_WISE.value,
        }
        self._check(XPU_DEVICES, sharding_types, (0, XPU_DEVICES - 1))

    @unittest.skipIf(XPU_DEVICES < 2, "requires two XPU devices")
    def test_column_wise_large_table_over_all_devices(self) -> None:
        if DIMENSION % (4 * XPU_DEVICES):
            self.skipTest("column shards must keep the dimension divisible by four")
        sharding_types = {
            "item_id": ShardingType.COLUMN_WISE.value,
            "user_id": ShardingType.TABLE_WISE.value,
            "item_category_id": ShardingType.TABLE_WISE.value,
        }
        self._check(XPU_DEVICES, sharding_types, (0,), ("boundaries", "mixed", "long"))

    @unittest.skipIf(XPU_DEVICES < 2, "requires two XPU devices")
    def test_sequence_outputs_to_each_device(self) -> None:
        generator = torch.Generator().manual_seed(7)
        expected = [
            torch.randn((5, DIMENSION), generator=generator),
            torch.randn((3, DIMENSION), generator=generator),
        ]
        for target_index in range(2):
            with self.subTest(target=target_index):
                target = torch.device(f"xpu:{target_index}")
                inputs = [
                    tensor.to(f"xpu:{index}") for index, tensor in enumerate(expected)
                ]
                outputs = SeqEmbeddingsAllToOne(target, world_size=2)(inputs)
                # The tensor already on the target is returned as it is.
                self.assertEqual(
                    outputs[target_index].data_ptr(), inputs[target_index].data_ptr()
                )
                for output, reference in zip(outputs, expected):
                    self.assertEqual(output.device, target)
                    torch.testing.assert_close(output.cpu(), reference)


if __name__ == "__main__":
    unittest.main()
