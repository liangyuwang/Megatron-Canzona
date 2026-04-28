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


def _comm_view(tensor):
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _validate_same_dtype_and_device(tensors):
    dtype = tensors[0].dtype
    device = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.dtype != dtype:
            raise TypeError(f"Expected all tensors to share dtype {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"Expected all tensors on {device}, got {tensor.device}")


class _FinalizeWork:
    """
    Wrap a ProcessGroup work handle and run a Python-side finalize step after wait().
    """

    def __init__(self, work, finalize):
        self._work = work
        self._finalize = finalize
        self._finalized = False

    def wait(self, *args, **kwargs):
        if self._work is not None:
            self._work.wait(*args, **kwargs)
        if not self._finalized:
            self._finalize()
            self._finalized = True
        return True

    def is_completed(self):
        if self._work is None:
            return self._finalized
        return self._work.is_completed()

    def __getattr__(self, name):
        return getattr(self._work, name)


class PaddedAllGatherPlan:
    """
    Two-phase padded all-gather-v helper.

    Call run() to enqueue the padded all_gather_into_tensor equivalent.
    Call finalize() after the communication has completed to unpack slices into
    the user-provided output tensors.

    This split makes the helper usable inside an outer _coalescing_manager(...):
      plan = prepare_padded_allgather(outputs, input)
      with _coalescing_manager(group, device=..., async_ops=...):
          plan.run(group, async_op=False)
      plan.finalize()
    """

    def __init__(self, outputs, input_tensor):
        if not outputs:
            raise ValueError("outputs must be a non-empty list")

        self.outputs = outputs
        self._output_views = [_comm_view(tensor) for tensor in outputs]
        self._input_view = _comm_view(input_tensor)
        _validate_same_dtype_and_device(self._output_views + [self._input_view])
        self.device = self._input_view.device
        self.launch_stream = None

        self.world_size = len(outputs)
        self.lengths = [tensor.numel() for tensor in self._output_views]
        self.max_numel = max(self.lengths)
        input_numel = self._input_view.numel()
        if input_numel > self.max_numel:
            raise ValueError(
                f"input numel {input_numel} exceeds max output numel {self.max_numel}"
            )

        self.padded_input = torch.zeros(
            self.max_numel, dtype=self._input_view.dtype, device=self.device
        )
        self.padded_output = torch.empty(
            self.world_size * self.max_numel,
            dtype=self._input_view.dtype,
            device=self.device,
        )

    def _stage_input(self):
        flat_input = self._input_view.contiguous().view(-1)
        self.padded_input[: flat_input.numel()].copy_(flat_input)

    def run(self, group, async_op=False):
        if group.size() != self.world_size:
            raise ValueError(f"outputs has {self.world_size} tensors, but group size is {group.size()}")
        rank = group.rank()
        if self._input_view.numel() != self.lengths[rank]:
            raise ValueError(
                f"input numel {self._input_view.numel()} does not match outputs[{rank}] "
                f"numel {self.lengths[rank]}"
            )
        if self.device.type == "cuda":
            self.launch_stream = torch.cuda.current_stream(self.device)
        self._stage_input()
        return torch.distributed.all_gather_into_tensor(
            self.padded_output,
            self.padded_input,
            group=group,
            async_op=async_op,
        )

    def finalize(self):
        def _copy():
            for rank, (output_view, length) in enumerate(zip(self._output_views, self.lengths)):
                chunk = self.padded_output.narrow(0, rank * self.max_numel, length)
                output_view.view(-1).copy_(chunk)

        if self.launch_stream is not None:
            with torch.cuda.stream(self.launch_stream):
                _copy()
        else:
            _copy()


class PaddedReduceScatterPlan:
    """
    Two-phase padded reduce-scatter-v helper.

    Call run() to enqueue the padded reduce_scatter_tensor equivalent.
    Call finalize() after the communication has completed to trim the padded
    result into the user-provided output tensor.
    """

    def __init__(self, output_tensor, inputs):
        if not inputs:
            raise ValueError("inputs must be a non-empty list")

        self.output_tensor = output_tensor
        self._output_view = _comm_view(output_tensor)
        self._input_views = [_comm_view(tensor) for tensor in inputs]
        _validate_same_dtype_and_device([self._output_view] + self._input_views)
        self.device = self._output_view.device
        self.launch_stream = None

        self.world_size = len(inputs)
        self.lengths = [tensor.numel() for tensor in self._input_views]
        self.max_numel = max(self.lengths)
        output_numel = self._output_view.numel()
        if output_numel > self.max_numel:
            raise ValueError(
                f"output numel {output_numel} exceeds max input numel {self.max_numel}"
            )

        self.padded_input = torch.zeros(
            self.world_size * self.max_numel,
            dtype=self._output_view.dtype,
            device=self.device,
        )
        self.padded_output = torch.empty(
            self.max_numel, dtype=self._output_view.dtype, device=self.device
        )

    def _stage_inputs(self):
        for rank, input_view in enumerate(self._input_views):
            flat_input = input_view.contiguous().view(-1)
            start = rank * self.max_numel
            self.padded_input[start : start + flat_input.numel()].copy_(flat_input)

    def run(self, group, reduce_op, async_op=False):
        if group.size() != self.world_size:
            raise ValueError(f"inputs has {self.world_size} tensors, but group size is {group.size()}")
        rank = group.rank()
        if self._output_view.numel() != self.lengths[rank]:
            raise ValueError(
                f"output numel {self._output_view.numel()} does not match inputs[{rank}] "
                f"numel {self.lengths[rank]}"
            )
        if self.device.type == "cuda":
            self.launch_stream = torch.cuda.current_stream(self.device)
        self._stage_inputs()
        return torch.distributed.reduce_scatter_tensor(
            self.padded_output,
            self.padded_input,
            op=reduce_op,
            group=group,
            async_op=async_op,
        )

    def finalize(self):
        def _copy():
            output_numel = self._output_view.numel()
            self._output_view.view(-1).copy_(self.padded_output[:output_numel])

        if self.launch_stream is not None:
            with torch.cuda.stream(self.launch_stream):
                _copy()
        else:
            _copy()


def prepare_padded_allgather(outputs, input_tensor):
    """
    Build a two-phase padded all-gather-v plan.
    """
    return PaddedAllGatherPlan(outputs, input_tensor)


def prepare_padded_reduce_scatter(output_tensor, inputs):
    """
    Build a two-phase padded reduce-scatter-v plan.
    """
    return PaddedReduceScatterPlan(output_tensor, inputs)


def padded_allgather(outputs, input_tensor, group, async_op=False):
    """
    Padded all-gather-v built on ProcessGroup::_allgather_base.

    This convenience wrapper is ideal for direct use. If you need to run inside
    an outer _coalescing_manager(...), use prepare_padded_allgather() + run() +
    finalize() so that finalize happens after the coalesced work completes.
    """
    plan = prepare_padded_allgather(outputs, input_tensor)
    work = plan.run(group, async_op=async_op)
    if async_op:
        return _FinalizeWork(work, plan.finalize)
    if work is not None:
        work.wait()
    plan.finalize()


def padded_reduce_scatter(output_tensor, inputs, group, reduce_op, async_op=False):
    """
    Padded reduce-scatter-v built on ProcessGroup::_reduce_scatter_base.

    This convenience wrapper is ideal for direct use. If you need to run inside
    an outer _coalescing_manager(...), use prepare_padded_reduce_scatter() +
    run() + finalize() so that finalize happens after the coalesced work
    completes.
    """
    plan = prepare_padded_reduce_scatter(output_tensor, inputs)
    work = plan.run(group, reduce_op, async_op=async_op)
    if async_op:
        return _FinalizeWork(work, plan.finalize)
    if work is not None:
        work.wait()
    plan.finalize()
