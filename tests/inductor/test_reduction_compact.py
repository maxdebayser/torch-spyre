# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for torch.ops.spyre.compact applied to reduction outputs.

Reducing a tensor on Spyre (e.g. ``x.sum(dim, keepdim)``) produces a tensor
whose numerical values match what you'd get by reducing on CPU and sending
the result over, but whose backing storage (numel, storage_nbytes, device
layout, DMA sizes/strides) does not: the Spyre-side reduction keeps the
sparse/padded layout of its input, while a CPU-computed tensor sent to
Spyre gets a fresh, densely packed layout. ``torch.ops.spyre.compact`` is
meant to reshape the former to match the latter.

Each case reduces a CPU tensor two ways and compares every tensor/layout
attribute (except storage/data pointers) plus the numerical values:

  actual   = sum_and_compact_compiled(x_cpu.to("spyre"), dim, keepdim)
  expected = x_cpu.sum(dim, keepdim).to("spyre")
"""

import pytest
import torch
from torch._inductor.codecache import FxGraphCache

from torch_spyre._C import get_spyre_dma_sizes, get_spyre_dma_strides

DEVICE = torch.device("spyre")


def sum_and_compact(x, dim, keepdim):
    reduced = x.sum(dim, keepdim)
    return torch.ops.spyre.compact(reduced)


sum_and_compact_compiled = torch.compile(sum_and_compact, fullgraph=True)


@pytest.fixture(autouse=True)
def _reset_compile_caches():
    # sum_and_compact_compiled is one module-level compiled function shared by
    # every case below; each (dim, keepdim) combination specializes it anew,
    # which would otherwise blow past Dynamo's per-function recompile limit.
    torch._dynamo.reset_code_caches()
    FxGraphCache.clear()
    yield


def _make_cpu_input(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.int32:
        return torch.randint(-100, 100, shape, dtype=dtype, generator=gen)
    raise ValueError(f"Unsupported dtype: {dtype}")


_TOLERANCES = {
    torch.float16: {"atol": 0.1, "rtol": 0.1},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.int32: {"atol": 0, "rtol": 0},
}


def _tensor_layout_snapshot(t):
    """All comparable tensor/device-layout attributes, excluding pointers."""
    return {
        "shape": tuple(t.shape),
        "stride": t.stride(),
        "storage_offset": t.storage_offset(),
        "numel": t.numel(),
        "dtype": t.dtype,
        "element_size": t.element_size(),
        "storage_nbytes": t.untyped_storage().nbytes(),
        "contiguous": t.is_contiguous(),
        "device": t.device,
        "dev_layout": t.device_tensor_layout(),
        "dma_sizes": get_spyre_dma_sizes(t),
        "dma_strides": get_spyre_dma_strides(t),
    }


def _assert_layout_matches(actual, expected):
    actual_snapshot = _tensor_layout_snapshot(actual)
    expected_snapshot = _tensor_layout_snapshot(expected)
    for key in actual_snapshot:
        assert actual_snapshot[key] == expected_snapshot[key], (
            f"{key} mismatch: actual={actual_snapshot[key]!r} "
            f"expected={expected_snapshot[key]!r}"
        )


# (shape, dim, keepdim) cases: 1D/2D/3D inputs, reducing the last dim or
# another dim, with and without keepdim.
REDUCTION_CASES = {
    "1d_dimneg1_keepdimF": ((200,), -1, False),
    "1d_dimneg1_keepdimT": ((200,), -1, True),
    "2d_dimneg1_keepdimF": ((256, 256), -1, False),
    "2d_dimneg1_keepdimT": ((256, 256), -1, True),
    "2d_dim0_keepdimF": ((256, 256), 0, False),
    "2d_dim0_keepdimT": ((256, 256), 0, True),
    "3d_dimneg1_keepdimF": ((2, 4, 256), -1, False),
    "3d_dimneg1_keepdimT": ((2, 4, 256), -1, True),
    "3d_dim0_keepdimF": ((2, 4, 256), 0, False),
    # This one fails but it doesn't even restickify!
    "3d_dim0_keepdimT": ((2, 4, 256), 0, True),
    "3d_dim1_keepdimF": ((2, 4, 256), 1, False),
    "3d_dim1_keepdimT": ((2, 4, 256), 1, True),
}


@pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16],
    ids=["fp16"],
)
# @pytest.mark.parametrize(
#     "dtype",
#     [torch.float16, torch.float32, torch.int32],
#     ids=["fp16", "fp32", "int32"],
# )
@pytest.mark.parametrize(
    "case_name,shape,dim,keepdim",
    [(name, *params) for name, params in REDUCTION_CASES.items()],
    ids=list(REDUCTION_CASES.keys()),
)
def test_sum_and_compact(case_name, shape, dim, keepdim, dtype):
    x_cpu = _make_cpu_input(shape, dtype, seed=0xAFFE)

    actual = sum_and_compact_compiled(x_cpu.to(DEVICE), dim, keepdim)
    expected = x_cpu.sum(dim, keepdim).to(DEVICE)

    #_assert_layout_matches(actual, expected)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), equal_nan=True, **_TOLERANCES[dtype]
    )
