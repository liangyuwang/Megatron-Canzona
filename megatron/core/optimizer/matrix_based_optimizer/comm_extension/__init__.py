import os
import torch
from torch.utils.cpp_extension import load

_src_dir = os.path.dirname(os.path.abspath(__file__))

_ext = load(
    name="coalesced_collectives",
    sources=[os.path.join(_src_dir, "coalesced_collectives.cpp")],
    extra_cflags=["-O3"],
    extra_include_paths=[
        os.path.dirname(torch.__file__) + "/include",
        os.path.dirname(torch.__file__) + "/include/torch/csrc/api/include",
    ],
    verbose=True,
)


def coalesced_allgather(outputs, input, group, async_op):
    """
    Simulate allgather(list) using broadcast loop.
    Does NOT nest startCoalescing/endCoalescing, so it is safe to call
    inside a _coalescing_manager(group, device=..., async_ops=...) block.

    outputs[rank] is expected to already contain the local data.
    input is explicitly passed as the root rank's data source for clarity.
    """
    _ext.allgather_coalesced(outputs, input, group, async_op)


def coalesced_reduce_scatter(output_tensor, inputs, group, reduce_op, async_op):
    """
    Simulate reduce_scatter(list) using reduce loop.
    Does NOT nest startCoalescing/endCoalescing, so it is safe to call
    inside a _coalescing_manager(group, device=..., async_ops=...) block.

    Note: output_tensor and inputs[rank] must share the same underlying memory.
    """
    _ext.reduce_scatter_coalesced(output_tensor, inputs, group, int(reduce_op), async_op)
