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

# Tests for torch.ops.spyre.compact — sparse-to-dense layout op.


from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch._dynamo as dynamo

import torch_spyre._inductor.passes as _passes
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout, get_spyre_dma_sizes, get_spyre_dma_strides
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


# -------- Tests: compact layout --------
#
# Tests that verify that the result of compact has the default
# STL for a tensor of a given torch shape

DTYPES = [torch.float16]
DTYPE_IDS = ["fp16"]

_TOLERANCES = {
    torch.float16: {"atol": 0.1, "rtol": 0.1},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.int32: {"atol": 0, "rtol": 0},
}


@torch.compile
def sum_and_compact(x, dim, keepdim):
    reduced = x.sum(dim, keepdim)
    return torch.ops.spyre.compact(reduced)


def _make_cpu_input(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    if dtype in (torch.float16, torch.float32):
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.int32:
        return torch.randint(-100, 100, shape, dtype=dtype, generator=gen)
    raise ValueError(f"Unsupported dtype: {dtype}")


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
    "3d_dim0_keepdimT": ((2, 4, 256), 0, True),
    "3d_dim1_keepdimF": ((2, 4, 256), 1, False),
    "3d_dim1_keepdimT": ((2, 4, 256), 1, True),
}


@pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
@pytest.mark.parametrize(
    "dtype",
    DTYPES,
    ids=DTYPE_IDS,
)
@pytest.mark.parametrize(
    "case_name,shape,dim,keepdim",
    [(name, *params) for name, params in REDUCTION_CASES.items()],
    ids=list(REDUCTION_CASES.keys()),
)
def test_sum_and_compact(case_name, shape, dim, keepdim, dtype):
    x_cpu = _make_cpu_input(shape, dtype, seed=0xAFFE)

    actual = sum_and_compact(x_cpu.to(DEVICE), dim, keepdim)
    expected = x_cpu.sum(dim, keepdim).to(DEVICE)

    _assert_layout_matches(actual, expected)
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), equal_nan=True, **_TOLERANCES[dtype]
    )


# -------- Tests: ops in compacted tensors --------
#
# Tests that verify that operation on "compacted" tensors run
# without crashing and produce correct results
filename = Path(__file__).stem

BOTH = [False, True]

_TOLERANCES = {
    torch.float16: {"atol": 0.1, "rtol": 0.1},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.int32: {"atol": 0, "rtol": 0},
}


def _ones(*args):
    return torch.ones(args)


# Allow in graph for debugging purposes
@torch.compiler.allow_in_graph
def maybe_compact(x: torch.Tensor, compact: bool):
    if compact and x.device.type == "spyre":
        return torch.ops.spyre.compact(x)
    return x


PRINT_FOR_REPRO = False


def run_binary_op(
    func, device, dtype, dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b
):
    if pre_op_keep_dim:
        # do this before sending to device to create the initial tensor layouts correctly
        b = b.unsqueeze(dim)
    a = a.to(device, dtype)
    b = b.to(device, dtype)

    if device == "cpu" and PRINT_FOR_REPRO:
        explanation = dynamo.explain(func)(
            dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b
        )
        for i, gm in enumerate(explanation.graphs):
            print(f"\nRepro {i}:\n")
            print("import torch")
            print("device='spyre'")
            print(f"a = torch.ones({tuple(a.shape)}, device=device, {dtype=})")
            print(f"b = torch.ones({tuple(b.shape)}, device=device, {dtype=})")
            print(
                f"{filename}_maybe_compact =  lambda x, _: torch.ops.spyre.compact(x)"
            )
            print(gm.code)
            print("compiled = torch.compile(forward)")
            print("print(compiled(None, a,b))")

    return func(dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b).cpu()


def run_test(do_run, compact):
    # run on CPU first to be sure that we didn't mess up the pytorch logic
    cpu_result = do_run("cpu")

    # Now that CPU hasn't failed, set xfail if we're not compacting
    if not compact:
        pytest.xfail("Operation may fail without compacting")

    spyre_result = do_run("spyre")

    torch.testing.assert_close(
        cpu_result, spyre_result, equal_nan=True, **_TOLERANCES[cpu_result.dtype]
    )


@torch.compile
def mul_on_reduced(dim, compact, reduce_keep_dim, pre_mul_keep_dim, a, b):
    reduced = a.sum(dim, keepdim=reduce_keep_dim)
    if reduce_keep_dim and not pre_mul_keep_dim:
        reduced.squeeze_(dim)
    elif not reduce_keep_dim and pre_mul_keep_dim:
        reduced.unsqueeze_(dim)
    reduced = maybe_compact(reduced, compact)
    return reduced * b


POINTWISE_CASES = {
    "scalar": (_ones(120), _ones(1), -1),
    "1d_stick": (_ones(128, 128), _ones(128), -1),
    "1d_nonstick": (_ones(128, 128), _ones(128), -2),
    "2d_stick": (_ones(2, 128, 128), _ones(2, 128), -1),
    "2d_nonstick": (_ones(2, 128, 128), _ones(2, 128), -1),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("compact", BOTH, ids=["no_compact", "compact"])
@pytest.mark.parametrize("reduce_keep_dim", BOTH)
@pytest.mark.parametrize("pre_mul_keep_dim", BOTH)
@pytest.mark.parametrize(
    "a,b,dim",
    list(POINTWISE_CASES.values()),
    ids=list(POINTWISE_CASES.keys()),
)
def test_pointwise_binary_op(
    dtype: torch.dtype,
    compact: bool,
    dim: int,
    reduce_keep_dim: bool,
    pre_mul_keep_dim: bool,
    a: torch.tensor,
    b: torch.tensor,
):
    def do_run(device):
        return run_binary_op(
            mul_on_reduced,
            device,
            dtype,
            dim,
            compact,
            reduce_keep_dim,
            pre_mul_keep_dim,
            a,
            b,
        )

    run_test(do_run, compact)


@torch.compile
def matmul_on_reduced(dim, compact, reduce_keep_dim, pre_mul_keep_dim, a, b):
    reduced = a.sum(dim, keepdim=reduce_keep_dim)
    if reduce_keep_dim and not pre_mul_keep_dim:
        reduced.squeeze_(dim)
    elif not reduce_keep_dim and pre_mul_keep_dim:
        reduced.unsqueeze_(dim)
    reduced = maybe_compact(reduced, compact)
    return reduced @ b


MATMUL_CASES = {
    "1d_stick": (_ones(128, 128), _ones(128), -1),
    "1d_nonstick": (_ones(128, 128), _ones(128), -2),
    "2d_stick": (_ones(2, 128, 128), _ones(128, 128), -1),
    "2d_nonstick": (_ones(2, 128, 128), _ones(128, 128), -2),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("compact", BOTH, ids=["no_compact", "compact"])
@pytest.mark.parametrize(
    "a,b,dim",
    list(MATMUL_CASES.values()),
    ids=list(MATMUL_CASES.keys()),
)
def test_matmul_op(
    dtype: torch.dtype, compact: bool, dim: int, a: torch.tensor, b: torch.tensor
):
    def do_run(device):
        return run_binary_op(
            matmul_on_reduced, device, dtype, dim, compact, False, False, a, b
        )

    run_test(do_run, compact)
