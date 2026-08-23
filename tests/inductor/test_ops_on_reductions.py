import pytest
import torch
import torch._dynamo as dynamo
from pathlib import Path

filename = Path(__file__).stem

DTYPES = [torch.float16]
DTYPE_IDS = ["fp16"]
#DTYPES=[torch.float16, torch.float32, torch.int32]
#DTYPE_IDS=["fp16", "fp32", "int32"]

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


def run_binary_op(func, device, dtype, dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b):
    if pre_op_keep_dim:
        # do this before sending to device to create the initial tensor layouts correctly
        b = b.unsqueeze(dim)
    a = a.to(device, dtype)
    b = b.to(device, dtype)

    if device == "cpu":
        explanation = dynamo.explain(func)(dim, compact, reduce_keep_dim, pre_op_keep_dim, a, b)
        for i, gm in enumerate(explanation.graphs):
            print(f"\nRepro {i}:\n")
            print("import torch")
            print("device='spyre'")
            print(f"a = torch.ones({tuple(a.shape)}, device=device, {dtype=})")
            print(f"b = torch.ones({tuple(b.shape)}, device=device, {dtype=})")
            print(f"{filename}_maybe_compact =  lambda x, _: torch.ops.spyre.compact(x)")
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
@pytest.mark.parametrize("compact", [True], ids=["compact"])
#@pytest.mark.parametrize("compact", BOTH, ids=["no_compact", "compact"])
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
    b: torch.tensor
):

    def do_run(device):
        return run_binary_op(mul_on_reduced, device, dtype, dim, compact,
                             reduce_keep_dim, pre_mul_keep_dim, a, b)

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
@pytest.mark.parametrize("compact", [True], ids=["compact"])
#@pytest.mark.parametrize("compact", BOTH, ids=["no_compact", "compact"])
@pytest.mark.parametrize(
    "a,b,dim",
    list(MATMUL_CASES.values()),
    ids=list(MATMUL_CASES.keys()),
)
def test_matmul_op(
    dtype: torch.dtype,
    compact: bool,
    dim: int,
    a: torch.tensor,
    b: torch.tensor
):

    def do_run(device):
        return run_binary_op(matmul_on_reduced, device, dtype, dim, compact,
                             False, False, a, b)

    run_test(do_run, compact)

