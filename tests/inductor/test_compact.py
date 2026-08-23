# Copyright 2025 The Torch-Spyre Authors.
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

# Tests for torch.ops.spyre.compact — sparse-to-dense layout reinterpret.
#
# compact converts a sparse Spyre tensor (one live element per stick,
# stride_map[-1] == -1) to a dense one via a zero-cost layout reinterpret:
# no restickify kernel is emitted between the reduction and the compact output.

from unittest.mock import patch

import pytest
import torch

import torch_spyre._inductor.passes as _passes
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout
from utils_inductor import _compile_and_run

DEVICE = torch.device("spyre")


# -------- Helpers --------


def _capture_plans(fn, args):
    """Run fn on DEVICE and capture restickify_plan."""
    captured = {}
    orig_finalize = _passes.finalize_layouts

    def capturing_finalize(graph):
        orig_finalize(graph)
        captured["restickify_plan"] = dict(V.graph.restickify_plan)

    with patch.object(_passes, "finalize_layouts", capturing_finalize):
        result = _compile_and_run(fn, args, DEVICE)

    return result, captured


def _assert_no_restickify(restickify_plan):
    assert not restickify_plan, (
        f"Expected no restickify, but got plan: {restickify_plan}"
    )


# -------- Tests: reduction-produced sparse input --------


def test_compact_reduction_sparse_correctness():
    """sum → compact → add: result matches expected value."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(192, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE).cpu()
    # sum over 128 ones = 128.0, doubled = 256.0
    expected = torch.full((192, 1), 256.0, dtype=torch.float16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)


def test_compact_reduction_sparse_no_restickify():
    """sum → compact → add: no restickify needed (sparse and dense (*S, 1) are byte-identical)."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(192, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    _, plans = _capture_plans(fn, [x_spyre])

    _assert_no_restickify(plans["restickify_plan"])


@pytest.mark.parametrize(
    "M,K",
    [
        (64, 128),  # M==elems_per_stick: regression for size-collision
        (64, 64),  # M==elems_per_stick: regression for size-collision
        (128, 128),
        (192, 128),
        (256, 128),
        (192, 64),
        (192, 256),
    ],
)
def test_compact_reduction_sparse_no_keepdim(M, K):
    """sum (no keepdim) → compact → add: 2D input, sparse 1D intermediate, dense 1D output."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=False)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(M, K, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE).cpu()
    expected = torch.full((M,), 2.0 * K, dtype=torch.float16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)


# -------- Tests: hand-constructed sparse input (produced inside fn) --------
# Sparse tensors can also enter a compiled graph as inputs (via an explicit
# device_layout=...). Tests for that path live further below; the tests here
# cover sparse intermediates produced inside fn.


def test_compact_hand_sparse_correctness():
    """Reduction along stick dim (dim=-1) produces sparse, compact → add: values correct."""

    def fn(x):
        # dim=-1 is the stick dim → reduction produces sparse layout
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(64, 192, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE).cpu()
    # sum over 192 ones = 192.0, doubled = 384.0
    expected = torch.full((64, 1), 384.0, dtype=torch.float16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)


def test_compact_hand_sparse_no_restickify():
    """Reduction along stick dim (dim=-1) produces sparse → compact → add: no restickify."""

    def fn(x):
        # dim=-1 is the stick dim → reduction produces sparse layout
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(64, 192, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    _, plans = _capture_plans(fn, [x_spyre])

    _assert_no_restickify(plans["restickify_plan"])


# -------- Tests: 3D tensors --------
#
# 3D input (batch, rows, cols): stick dim is the last dim (cols).
# Only sum(dim=-1) produces a sparse layout; sum(dim=0) and sum(dim=1) are dense.
#
# Verified layouts for input (4, 48, 128) fp16:
#   input:                device_size=[48, 2, 4, 64]   stride_map=[128, 64, 6144, 1]
#   sum(dim=-1,kd=True):  device_size=[48, 1, 1, 4, 64] stride_map=[1, -1, -1, 48, -1]  ← sparse
#   after compact:        device_size=[48, 1, 4, 64]   stride_map=[1, -1, 48, -1]        ← dense


def test_compact_3d_correctness():
    """3D: sum(dim=-1) → compact → add: result matches expected value."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE).cpu()
    # sum over 128 ones = 128.0, doubled = 256.0
    expected = torch.full((4, 48, 1), 256.0, dtype=torch.float16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)


def test_compact_3d_no_restickify():
    """3D: sum(dim=-1) → compact → add: no restickify needed (byte-identical)."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    _, plans = _capture_plans(fn, [x_spyre])

    _assert_no_restickify(plans["restickify_plan"])


def test_compact_3d_layouts():
    """3D: verify sparse→dense device_size and stride_map transformation."""

    captured = {}
    orig_finalize = _passes.finalize_layouts

    def capturing_finalize(graph):
        orig_finalize(graph)
        for name, buf in graph.name_to_buffer.items():
            try:
                layout = buf.get_layout()
                if hasattr(layout, "device_layout"):
                    captured.setdefault("layouts", {})[name] = layout.device_layout
            except Exception:
                pass

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)

    with patch.object(_passes, "finalize_layouts", capturing_finalize):
        _compile_and_run(fn, [x_spyre], DEVICE)

    layouts = captured["layouts"]
    # buf0: sum output — sparse, device_rank 5
    # buf1: compact output — dense, device_rank 4 (one dim stripped)
    buf0_stl = layouts["buf0"]
    buf1_stl = layouts["buf1"]

    # Sparse: two inner size-1 dims (tile count + outer-stick) both synthetic
    assert list(buf0_stl.device_size) == [48, 1, 1, 4, 64], (
        f"Unexpected sparse device_size: {list(buf0_stl.device_size)}"
    )
    assert list(buf0_stl.stride_map) == [1, -1, -1, 48, -1], (
        f"Unexpected sparse stride_map: {list(buf0_stl.stride_map)}"
    )

    # Dense: one dim fewer, stick dim is now real (stride_map[-1] != -1 except
    # stick synthetic marker; verify it's the standard dense (4,48,1) layout)
    assert list(buf1_stl.device_size) == [48, 1, 4, 64], (
        f"Unexpected dense device_size: {list(buf1_stl.device_size)}"
    )
    assert list(buf1_stl.stride_map) == [1, -1, 48, -1], (
        f"Unexpected dense stride_map: {list(buf1_stl.stride_map)}"
    )



# -------- Tests: keepdim=False (3D input → 2D sparse → compact → 2D dense) --------
#
# All dim sizes must be multiples of 64 (stick size for fp16) due to a known
# limitation in transpose/restickify for non-aligned sizes.


def test_compact_keepdim_false_3d_correctness():
    """3D sum(dim=-1, keepdim=False) → compact → add: result matches expected value."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=False)
        y = torch.ops.spyre.compact(y)
        return y + y

    # (4, 128, 64): all dims divisible by 64; M=128 avoids the M==64 ambiguity
    x = torch.ones(4, 128, 64, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE).cpu()
    # sum over 64 ones = 64.0, doubled = 128.0
    expected = torch.full((4, 128), 128.0, dtype=torch.float16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)


# -------- Tests: sparse graph-input feeding compact --------
#
# A sparse tensor can be passed in as a graph input by constructing it on the
# host with an explicit device_layout=SpyreTensorLayout(..., dim_order).

def _sparse_graph_input(
    host_tensor: torch.Tensor, dim_order: list[int]
) -> torch.Tensor:
    # Warm the spyre runtime so torch.Tensor.to is monkey-patched to accept
    # device_layout=. The patch is installed lazily on first contact with the
    # spyre device.
    torch.empty(1, dtype=torch.float16).to(DEVICE)
    stl = SpyreTensorLayout(
        host_tensor.size(), host_tensor.stride(), host_tensor.dtype, dim_order
    )
    return host_tensor.to(DEVICE, device_layout=stl)


def test_compact_sparse_graph_input_canonical_correctness():
    """Sparse graph input (trailing 1) with default dim_order → compact: values correct."""

    def fn(t):
        y = torch.ops.spyre.compact(t)
        return y + y

    x = torch.arange(192, dtype=torch.float16).reshape(192, 1)
    x_sparse = _sparse_graph_input(x, [0, 1, -1])
    result = _compile_and_run(fn, [x_sparse], DEVICE).cpu()
    torch.testing.assert_close(result, 2.0 * x, rtol=0, atol=0)


def test_compact_sparse_graph_input_non_canonical_correctness():
    """Sparse graph input (trailing 1) with non-default dim_order → compact: values correct."""

    def fn(t):
        y = torch.ops.spyre.compact(t)
        return y + y

    x = torch.arange(192, dtype=torch.float16).reshape(192, 1)
    x_sparse = _sparse_graph_input(x, [1, 0, -1])
    result = _compile_and_run(fn, [x_sparse], DEVICE).cpu()
    torch.testing.assert_close(result, 2.0 * x, rtol=0, atol=0)
