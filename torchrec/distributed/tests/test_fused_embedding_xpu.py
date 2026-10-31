#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Fused (training) embedding tables and the training pipeline on one Intel XPU.

The tests run on XPU with an XPU build of the FBGEMM operators (for example the
fbgemm-xpu plugin from intel/torchlib-xpu) and an fbgemm_gpu whose
ComputeDevice has XPU, and are skipped elsewhere. They compare training steps
with the same steps computed on the CPU, and TrainPipelineSparseDist with a
plain training loop.
"""

import importlib
import importlib.util
import socket
import unittest
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.types import ShardingPlan, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

ROWS = 1000
DIMENSION = 64
# Ids stay below IDS, so every batch looks up some rows more than once.
IDS = 32
LEARNING_RATE = 0.1
EPS = 1e-8


def _xpu_training_available() -> bool:
    if not torch.xpu.is_available() or importlib.util.find_spec("fbgemm_xpu") is None:
        return False
    importlib.import_module("fbgemm_xpu")
    return hasattr(ComputeDevice, "XPU")


XPU_TRAINING: bool = _xpu_training_available()


def _tables() -> List[EmbeddingConfig]:
    return [
        EmbeddingConfig(
            name="t0",
            embedding_dim=DIMENSION,
            num_embeddings=ROWS,
            feature_names=["f0"],
        )
    ]


def _sharder() -> EmbeddingCollectionSharder:
    return EmbeddingCollectionSharder(
        fused_params={
            "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            "learning_rate": LEARNING_RATE,
            "eps": EPS,
        }
    )


def _plan(module: nn.Module, sharding_type: str) -> ShardingPlan:
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            world_size=1, local_world_size=1, compute_device="xpu", hbm_cap=2**30
        ),
        constraints={
            "t0": ParameterConstraints(
                sharding_types=[sharding_type], compute_kernels=["fused"]
            )
        },
    )
    # pyrefly: ignore[bad-argument-type]
    return planner.plan(module, [_sharder()])


def _features(seed: int, empty: bool = False) -> KeyedJaggedTensor:
    generator = torch.Generator().manual_seed(seed)
    lengths = (
        torch.zeros(8, dtype=torch.int64)
        if empty
        else torch.randint(1, 5, (8,), generator=generator)
    )
    values = torch.randint(0, IDS, (int(lengths.sum()),), generator=generator)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=["f0"], values=values, lengths=lengths
    )


def _full_weight(module: nn.Module) -> torch.Tensor:
    state = module.state_dict()
    tensor = state[next(key for key in state if key.endswith("t0.weight"))]
    if not hasattr(tensor, "local_shards"):
        return tensor.detach().cpu()
    weight = torch.zeros(ROWS, DIMENSION)
    for shard in tensor.local_shards():
        row, column = shard.metadata.shard_offsets
        rows, columns = shard.tensor.shape
        weight[row : row + rows, column : column + columns] = (
            shard.tensor.detach().cpu()
        )
    return weight


def _tbes(dmp: DistributedModelParallel) -> List[SplitTableBatchedEmbeddingBagsCodegen]:
    # The sharded module keeps its lookups in a plain list.
    tbes = []
    for lookup in getattr(dmp._dmp_wrapped_module, "_lookups", []):
        for module in getattr(lookup, "module", lookup).modules():
            if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
                tbes.append(module)
    return tbes


class _TrainModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.ec = EmbeddingCollection(tables=_tables(), device=torch.device("meta"))
        self.over = nn.Linear(DIMENSION, 1, device=device)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.ec(batch.sparse_features)["f0"].values()
        prediction = self.over(values).sum()
        return (prediction - batch.labels.sum()) ** 2, prediction.detach()


@unittest.skipIf(
    not XPU_TRAINING,
    "requires an XPU device, fbgemm-xpu and an fbgemm_gpu with ComputeDevice.XPU",
)
class FusedEmbeddingXpuTest(unittest.TestCase):
    device: torch.device = torch.device("xpu", 0)

    @classmethod
    def setUpClass(cls) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "xccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
        torch.xpu.set_device(cls.device)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()

    def _dmp(self, module: nn.Module, sharding_type: str) -> DistributedModelParallel:
        return DistributedModelParallel(
            module=module,
            device=self.device,
            plan=_plan(module, sharding_type),
            # pyrefly: ignore[bad-argument-type]
            sharders=[_sharder()],
            init_data_parallel=False,
        )

    def test_fused_steps_match_cpu(self) -> None:
        for sharding_type in (
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ):
            with self.subTest(sharding_type=sharding_type):
                torch.manual_seed(0)
                dmp = self._dmp(
                    EmbeddingCollection(tables=_tables(), device=torch.device("meta")),
                    sharding_type,
                )
                tbes = _tbes(dmp)
                self.assertTrue(tbes)
                for tbe in tbes:
                    self.assertFalse(tbe.use_cpu)
                    self.assertEqual(tbe.weights_dev.device.type, "xpu")
                    self.assertTrue(
                        all(spec[3].name == "XPU" for spec in tbe.embedding_specs)
                    )

                weight = _full_weight(dmp)
                momentum = torch.zeros(ROWS)
                for step in range(2):
                    features = _features(seed=step + 1)
                    output = dmp(features.to(self.device))
                    if hasattr(output, "wait"):
                        output = output.wait()
                    embeddings = output["f0"].values()
                    torch.testing.assert_close(
                        embeddings.detach().cpu(),
                        weight[features.values()],
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    grad = torch.randn(
                        embeddings.shape, generator=torch.Generator().manual_seed(9)
                    )
                    (embeddings * grad.to(self.device)).sum().backward()
                    torch.xpu.synchronize()

                    # Exact row-wise Adagrad on the CPU, from the same gradient.
                    row_grad = torch.zeros(ROWS, DIMENSION).index_add_(
                        0, features.values(), grad
                    )
                    rows = torch.unique(features.values())
                    momentum[rows] += (row_grad[rows] ** 2).mean(dim=1)
                    weight[rows] -= (
                        LEARNING_RATE
                        * row_grad[rows]
                        / (momentum[rows].sqrt() + EPS).unsqueeze(1)
                    )
                    torch.testing.assert_close(
                        _full_weight(dmp), weight, rtol=1e-5, atol=1e-6
                    )
                    states = [
                        state[0].detach().cpu()
                        for tbe in tbes
                        for state in tbe.split_optimizer_states()
                    ]
                    for state in states:
                        torch.testing.assert_close(
                            state.flatten(), momentum, rtol=1e-5, atol=1e-6
                        )

    def test_pipeline_matches_plain_loop(self) -> None:
        def build() -> Tuple[DistributedModelParallel, KeyedOptimizerWrapper]:
            torch.manual_seed(0)
            model = _TrainModel(self.device)
            dmp = self._dmp(model, ShardingType.ROW_WISE.value)
            optimizer = KeyedOptimizerWrapper(
                dict(in_backward_optimizer_filter(dmp.named_parameters())),
                lambda params: torch.optim.SGD(params, lr=0.01),
            )
            return dmp, optimizer

        # The fourth batch has no ids at all.
        batches = [
            Batch(
                dense_features=torch.zeros(8, 1),
                sparse_features=_features(seed=10 + index, empty=index == 3),
                labels=torch.ones(8),
            )
            for index in range(6)
        ]
        plain, plain_optimizer = build()
        piped, piped_optimizer = build()
        # pyrefly: ignore[bad-argument-type]
        piped.load_state_dict(plain.state_dict())

        expected = []
        for batch in batches:
            loss, prediction = plain(batch.to(self.device))
            plain_optimizer.zero_grad()
            loss.backward()
            plain_optimizer.step()
            expected.append(prediction.cpu())

        pipeline = TrainPipelineSparseDist(
            model=piped, optimizer=piped_optimizer, device=self.device
        )
        memcpy_stream: Optional[torch.Stream] = pipeline._memcpy_stream
        self.assertIsNotNone(memcpy_stream)
        self.assertEqual(memcpy_stream.device.type, "xpu")

        actual = []
        batch_iterator = iter(batches)
        while True:
            try:
                actual.append(pipeline.progress(batch_iterator).cpu())
            except StopIteration:
                break
        # Every batch comes out once the iterator is exhausted.
        self.assertEqual(len(actual), len(batches))
        for got, reference in zip(actual, expected):
            torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            _full_weight(piped), _full_weight(plain), rtol=1e-5, atol=1e-6
        )
        with self.assertRaises(StopIteration):
            pipeline.progress(batch_iterator)


if __name__ == "__main__":
    unittest.main()
