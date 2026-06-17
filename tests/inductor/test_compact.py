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
    """Run fn on DEVICE and capture restickify_plan and reinterpret_plan."""
    captured = {}
    orig_finalize = _passes.finalize_layouts

    def capturing_finalize(graph):
        orig_finalize(graph)
        captured["restickify_plan"] = dict(V.graph.restickify_plan)
        captured["reinterpret_plan"] = dict(getattr(V.graph, "reinterpret_plan", {}))

    with patch.object(_passes, "finalize_layouts", capturing_finalize):
        result = _compile_and_run(fn, args, DEVICE)

    return result, captured


def _assert_no_restickify(restickify_plan):
    assert not restickify_plan, (
        f"Expected no restickify, but got plan: {restickify_plan}"
    )


def _assert_has_reinterpret(reinterpret_plan):
    assert reinterpret_plan, (
        "Expected a reinterpret entry, but reinterpret_plan is empty"
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
    """sum → compact → add: no restickify kernel, one reinterpret entry."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(192, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    _, plans = _capture_plans(fn, [x_spyre])

    _assert_no_restickify(plans["restickify_plan"])
    _assert_has_reinterpret(plans["reinterpret_plan"])


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
    _assert_has_reinterpret(plans["reinterpret_plan"])


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
    """3D: sum(dim=-1) → compact → add: no restickify, one reinterpret entry."""

    def fn(x):
        y = torch.sum(x, dim=-1, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    _, plans = _capture_plans(fn, [x_spyre])

    _assert_no_restickify(plans["restickify_plan"])
    _assert_has_reinterpret(plans["reinterpret_plan"])


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


# -------- Negative test: dense input should not reinterpret --------


def test_compact_dense_input_rejected():
    """compact on a dense input (non-stick-dim reduction) raises an error.

    Previously this silently miscompiled: the keepdim=False chain's
    reinterpret kernel produced 64 copies of one input element per stick
    instead of the original 64 distinct values.  The propagator now rejects
    dense input to spyre::reinterpret so the failure is loud rather than
    silent.
    """

    def fn(x):
        # sum over dim=0 (non-stick dim) produces dense output — invalid for compact.
        y = torch.sum(x, dim=0, keepdim=True)
        y = torch.ops.spyre.compact(y)
        return y + y

    x = torch.ones(64, 192, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    with pytest.raises(Exception, match="sparse input"):
        _compile_and_run(fn, [x_spyre], DEVICE)


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


# -------- spyre::reinterpret standalone tests --------


def test_reinterpret_2d_shape():
    """reinterpret on sparse (4, 48) → output shape is (4, 48, 64)."""

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)
    assert list(result.shape) == [4, 48, 64], f"Unexpected shape: {result.shape}"


def test_reinterpret_2d_no_crash():
    """reinterpret on sparse (4, 48) compiles and produces correct values."""

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    x = torch.ones(4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)
    assert list(result.shape) == [4, 48, 64], f"Unexpected shape: {result.shape}"
    assert (result == 128.0).all(), f"Expected all 128.0, got {result.unique()}"


def test_reinterpret_2d_values_arange():
    """reinterpret: per-row sum value is broadcast across all 64 stick slots.

    x[b, m, :] = [m, m, ..., m] (row value is m), so sum = 128*m.
    After reinterpret, result[b, m, i] == 128*m for all i in 0..63.
    """

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    # shape (4, 48, 128): row (b, m) has all values == m
    row_vals = (
        torch.arange(48, dtype=torch.float16).reshape(1, 48, 1).expand(4, 48, 128)
    )
    x = row_vals.contiguous()
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)

    assert list(result.shape) == [4, 48, 64], f"Unexpected shape: {result.shape}"
    # expected[b, m, i] = 128 * m for all b, i
    expected = (
        (128.0 * torch.arange(48, dtype=torch.float16))
        .reshape(1, 48, 1)
        .expand(4, 48, 64)
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "B,M,K",
    [
        (1, 48, 128),  # batch=1
        (8, 192, 64),  # larger batch and rows, K=64 (one stick)
        (2, 96, 256),  # wider reduction dim
        (4, 64, 128),  # M==elems_per_stick: regression for size-collision
    ],
)
def test_reinterpret_shapes(B, M, K):
    """reinterpret: correct output shape and uniform values for various 3D input shapes."""

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    x = torch.ones(B, M, K, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)

    elems = 64  # fp16 elems per stick
    assert list(result.shape) == [B, M, elems], f"Unexpected shape: {result.shape}"
    assert (result == float(K)).all(), f"Expected all {float(K)}, got {result.unique()}"


def test_reinterpret_4d_values():
    """reinterpret on 4D sparse (B, D, M): output shape (B, D, M, 64) with correct values."""

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    x = torch.ones(2, 4, 48, 128, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)

    assert list(result.shape) == [2, 4, 48, 64], f"Unexpected shape: {result.shape}"
    assert (result == 128.0).all(), f"Expected all 128.0, got {result.unique()}"


def test_reinterpret_4d_values_arange():
    """reinterpret on 4D input: per-row sum broadcast across all 64 stick slots.

    x[b, d, m, :] = m, so sum = 128*m.
    After reinterpret, result[b, d, m, i] == 128*m for all i in 0..63.
    """

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    row_vals = (
        torch.arange(48, dtype=torch.float16).reshape(1, 1, 48, 1).expand(2, 4, 48, 128)
    )
    x = row_vals.contiguous()
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)

    assert list(result.shape) == [2, 4, 48, 64], f"Unexpected shape: {result.shape}"
    expected = (
        (128.0 * torch.arange(48, dtype=torch.float16))
        .reshape(1, 1, 48, 1)
        .expand(2, 4, 48, 64)
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "B,D,M,K",
    [
        (1, 2, 96, 64),  # batch=1, K=64 (one stick)
        (4, 2, 48, 128),  # standard 4D
        (2, 8, 48, 64),  # more depth, K=64
        (2, 4, 64, 128),  # M==elems_per_stick: regression for size-collision
    ],
)
def test_reinterpret_4d_shapes(B, D, M, K):
    """reinterpret: correct output shape and values for various 4D input shapes."""

    def fn(x):
        sparse = torch.sum(x, dim=-1, keepdim=False)
        return torch.ops.spyre.reinterpret(sparse)

    x = torch.ones(B, D, M, K, dtype=torch.float16)
    x_spyre = x.to(DEVICE)
    result = _compile_and_run(fn, [x_spyre], DEVICE)

    elems = 64
    assert list(result.shape) == [B, D, M, elems], f"Unexpected shape: {result.shape}"
    assert (result == float(K)).all(), f"Expected all {float(K)}, got {result.unique()}"


# -------- Tests: sparse graph-input feeding compact --------
#
# A sparse tensor can be passed in as a graph input by constructing it on the
# host with an explicit device_layout=SpyreTensorLayout(..., dim_order). The
# input shape (N, 1) (trailing size-1 dim) takes the keepdim=True branch of
# compact_decomp, which delegates to spyre::compact_relabel — that's the path
# where REINTERPRET-(1) fires for canonical sparse inputs. Two subcases:
#   - default dim_order [0, 1, ..., rank-1, -1]: the canonical sparse layout
#     produced by reductions; REINTERPRET-(1) fires (free relabel) and the
#     TensorBox unwrap in _apply_reinterpret_on_producer is exercised.
#   - non-default dim_order: REINTERPRET-(1) precondition fails (correctly,
#     since relabeling across dim_orders would reorder host elements);
#     restickify is inserted to materialize the canonical sparse layout
#     before compact_relabel runs.


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
    """Sparse graph input (trailing 1) with default dim_order → compact: free reinterpret path."""

    def fn(t):
        y = torch.ops.spyre.compact(t)
        return y + y

    x = torch.arange(192, dtype=torch.float16).reshape(192, 1)
    x_sparse = _sparse_graph_input(x, [0, 1, -1])
    result = _compile_and_run(fn, [x_sparse], DEVICE).cpu()
    torch.testing.assert_close(result, 2.0 * x, rtol=0, atol=0)


def test_compact_sparse_graph_input_canonical_no_restickify():
    """Sparse graph input with default dim_order → compact: reinterpret entry, no restickify.

    Exercises the TensorBox unwrap in _apply_reinterpret_on_producer — without
    that fix, this path crashes when the optimizer tries to set .layout on a
    TensorBox-wrapped graph input.
    """

    def fn(t):
        y = torch.ops.spyre.compact(t)
        return y + y

    x = torch.arange(192, dtype=torch.float16).reshape(192, 1)
    x_sparse = _sparse_graph_input(x, [0, 1, -1])
    _, plans = _capture_plans(fn, [x_sparse])

    _assert_no_restickify(plans["restickify_plan"])
    _assert_has_reinterpret(plans["reinterpret_plan"])


def test_compact_sparse_graph_input_non_canonical_correctness():
    """Sparse graph input (trailing 1) with non-default dim_order → compact: restickify path."""

    def fn(t):
        y = torch.ops.spyre.compact(t)
        return y + y

    x = torch.arange(192, dtype=torch.float16).reshape(192, 1)
    x_sparse = _sparse_graph_input(x, [1, 0, -1])
    result = _compile_and_run(fn, [x_sparse], DEVICE).cpu()
    torch.testing.assert_close(result, 2.0 * x, rtol=0, atol=0)


def test_compact_sparse_graph_input_non_canonical_skips_reinterpret():
    """Sparse graph input with non-default dim_order → compact: REINTERPRET-(1) skipped.

    REINTERPRET-(1) requires default dim_order on the producer; with a
    non-canonical sparse input, the precondition must fail. The optimizer
    falls back to whatever path the consumer needs (restickify or a
    layout-aware lowering); the key invariant is that the unsound free
    relabel does NOT fire.
    """

    def fn(t):
        y = torch.ops.spyre.compact(t)
        return y + y

    x = torch.arange(192, dtype=torch.float16).reshape(192, 1)
    x_sparse = _sparse_graph_input(x, [1, 0, -1])
    _, plans = _capture_plans(fn, [x_sparse])

    assert not plans["reinterpret_plan"], (
        f"Expected no reinterpret for non-canonical sparse input, got: {plans['reinterpret_plan']}"
    )
