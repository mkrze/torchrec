#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Dict, Optional

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

MAX_SIZE: int = (1 << 63) - 1

INTRA_NODE_BANDWIDTH: float = 600 * 1024 * 1024 * 1024 / 1000  # bytes/ms
CROSS_NODE_BANDWIDTH: float = 12.5 * 1024 * 1024 * 1024 / 1000  # bytes/ms

MIN_CW_DIM: int = 128
POOLING_FACTOR: float = 1.0
NUM_POOLINGS: float = 1.0

BIGINT_DTYPE: int = 8

HBM_CAP: int = 32 * 1024 * 1024 * 1024  # 32 GB
DDR_CAP: int = 128 * 1024 * 1024 * 1024  # 128 GB
SSD_CAP: int = 2 * 1024 * 1024 * 1024 * 1024  # 2 TB
DDR_MEM_BW: float = 51 * 1024 * 1024 * 1024 / 1000  # bytes/ms
HBM_MEM_BW: float = 897 * 1024 * 1024 * 1024 / 1000  # bytes/ms
SSD_MEM_BW: float = 100 * 1024 * 1024 / 1000  # bytes/ms
# This can be smaller than DDR_MEM_BW because the PCI channel maybe shared
# with other devices such as the FE NIC.
HBM_TO_DDR_MEM_BW: float = 32 * 1024 * 1024 * 1024 / 1000  # bytes/ms
UVM_CACHING_RATIO: float = 0.2
KV_CACHING_RATIO: float = 0.2
BATCH_SIZE: int = 512

BATCHED_COPY_PERF_FACTOR: float = 2.455  # empirical studies
FULL_BLOCK_EMB_DIM: int = 128  # FBGEMM Kernel, 32 threads X 4D-Vector
HALF_BLOCK_PENALTY: float = 1.15  # empirical studies
QUARTER_BLOCK_PENALTY: float = 1.75  # empirical studies
BWD_COMPUTE_MULTIPLIER: float = 2  # empirical studies
WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER: float = 1  # empirical studies
WEIGHTED_KERNEL_MULTIPLIER: float = 1.1  # empirical studies
DP_ELEMENTWISE_KERNELS_PERF_FACTOR: float = 9.22  # empirical studies

# For rounding memory hashing to nearest 100GB (10^11 bytes):
HUNDRED_GB = 100 * 1024**3  # 107,374,182,400

# Default perf estimator name - use this constant when creating estimators
DEFAULT_PERF_ESTIMATOR: str = "default_estimator"

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-generic HBM bandwidth lookup (bytes/ms).
# Maps device name substrings to measured/theoretical HBM bandwidth.
# Extend this dict when adding support for new GPU families.
#
# Values are PER TORCH DEVICE, i.e. per stack, not per card. The planner builds
# a Topology of world_size devices and applies this figure to each one, so a
# whole-card number would overstate bandwidth on multi-stack parts. A Max 1550
# under the default FLAT device hierarchy enumerates as two torch devices of one
# stack each, so it takes the same per-stack figure as the single-stack 1100.
# ---------------------------------------------------------------------------
_XPU_HBM_MEM_BW: Dict[str, float] = {
    # Intel Data Center GPU Max 1100: 1 stack, 48 GB HBM2e, ~614 GB/s
    "Max 1100": 614 * 1024 * 1024 * 1024 / 1000,
    # Intel Data Center GPU Max 1550: 2 stacks, 64 GB and ~614 GB/s each
    "Max 1550": 614 * 1024 * 1024 * 1024 / 1000,
}

# Default XPU HBM bandwidth when device name is not recognized.
_XPU_HBM_MEM_BW_DEFAULT: float = 614 * 1024 * 1024 * 1024 / 1000  # bytes/ms


def detect_hbm_mem_bw(
    compute_device: str,
    device_index: int = 0,
) -> float:
    """Detect HBM memory bandwidth (bytes/ms) for the given device.

    For XPU devices, looks up bandwidth by device name.
    For CUDA devices, returns the default HBM_MEM_BW constant.
    Falls back to HBM_MEM_BW for unknown device types.
    """
    if compute_device == "xpu" and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(device_index)
        device_name = props.name
        for key, bw in _XPU_HBM_MEM_BW.items():
            if key in device_name:
                logger.info(
                    f"XPU device '{device_name}' matched bandwidth profile "
                    f"'{key}': {bw * 1000 / (1024**3):.0f} GB/s"
                )
                return bw
        logger.warning(
            f"XPU device '{device_name}' not in bandwidth table, "
            f"using default {_XPU_HBM_MEM_BW_DEFAULT * 1000 / (1024**3):.0f} GB/s"
        )
        return _XPU_HBM_MEM_BW_DEFAULT
    return HBM_MEM_BW


def detect_hbm_cap(
    compute_device: str,
    device_index: int = 0,
) -> int:
    """Detect device HBM capacity in bytes.

    Supports CUDA and XPU devices.  Returns 0 for CPU.
    """
    if compute_device == "xpu" and torch.xpu.is_available():
        return torch.xpu.get_device_properties(device_index).total_memory
    elif compute_device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_index).total_mem
    return 0


def kernel_bw_lookup(
    compute_device: str,
    compute_kernel: str,
    hbm_mem_bw: float,
    ddr_mem_bw: float,
    hbm_to_ddr_mem_bw: float,
    caching_ratio: Optional[float] = None,
    prefetch_pipeline: bool = False,
    ssd_mem_bw: float = SSD_MEM_BW,
) -> Optional[float]:
    """
    Calculates the device bandwidth based on given compute device, compute kernel, and
    caching ratio.

    Args:
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        hbm_mem_bw (float): the bandwidth of the device HBM.
        ddr_mem_bw (float): the bandwidth of the system DDR memory.
        hbm_to_ddr_bw (float): the bandwidth between device HBM and system DDR.
        caching_ratio (Optional[float]): caching ratio used to determine device bandwidth
            if UVM caching is enabled.
        prefetch_pipeline (bool): whether prefetch pipeline is enabled.
        ssd_mem_bw (float): the bandwidth of SSD storage. Added at the end of the
            parameter list (with a default value) to maintain backward compatibility
            with existing callers.

    Returns:
        Optional[float]: the device bandwidth.
    """
    caching_ratio = caching_ratio if caching_ratio is not None else UVM_CACHING_RATIO
    lookup = {
        # CPU
        ("cpu", EmbeddingComputeKernel.DENSE.value): 0.5 * ddr_mem_bw,
        ("cpu", EmbeddingComputeKernel.FUSED.value): 1 * ddr_mem_bw,
        ("cpu", EmbeddingComputeKernel.QUANT.value): 1 * ddr_mem_bw,
        # TODO: Determine the correct value later. MTIA uses values same as CPU's.
        # MTIA
        ("mtia", EmbeddingComputeKernel.DENSE.value): 0.5 * hbm_mem_bw,
        ("mtia", EmbeddingComputeKernel.FUSED.value): 1 * hbm_mem_bw,
        ("mtia", EmbeddingComputeKernel.QUANT.value): 1 * hbm_mem_bw,
        # CUDA
        ("cuda", EmbeddingComputeKernel.DENSE.value): 0.5 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED_TRITON.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED_UVM.value): hbm_to_ddr_mem_bw / 10,
        ("cuda", EmbeddingComputeKernel.FUSED_UVM_CACHING.value): (
            caching_ratio * hbm_mem_bw + (1 - caching_ratio) * hbm_to_ddr_mem_bw
        )
        / 10,
        ("cuda", EmbeddingComputeKernel.QUANT.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.QUANT_UVM.value): hbm_to_ddr_mem_bw / 10,
        ("cuda", EmbeddingComputeKernel.QUANT_UVM_CACHING.value): (
            caching_ratio * hbm_mem_bw + (1 - caching_ratio) * hbm_to_ddr_mem_bw
        )
        / 10,
        # TODO: revisit whether this estimation makes sense
        ("cuda", EmbeddingComputeKernel.KEY_VALUE.value): ssd_mem_bw,
        ("cuda", EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value): hbm_to_ddr_mem_bw,
        ("cuda", EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value): hbm_to_ddr_mem_bw,
        (
            "cuda",
            EmbeddingComputeKernel.DRAM_SSD_VIRTUAL_TABLE.value,
        ): hbm_to_ddr_mem_bw,
        # TODO: revisit update the values
        ("tpu", EmbeddingComputeKernel.DENSE.value): 0.5 * hbm_mem_bw,
        ("tpu", EmbeddingComputeKernel.FUSED.value): 1 * hbm_mem_bw,
        ("tpu", EmbeddingComputeKernel.QUANT.value): 1 * hbm_mem_bw,
        ("xpu", EmbeddingComputeKernel.DENSE.value): 0.5 * hbm_mem_bw,
        ("xpu", EmbeddingComputeKernel.FUSED.value): 1 * hbm_mem_bw,
    }

    if (
        prefetch_pipeline
        and compute_device in ("cuda", "xpu")
        and compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
    ):
        return lookup.get(("cuda", EmbeddingComputeKernel.FUSED.value))

    return lookup.get((compute_device, compute_kernel))
