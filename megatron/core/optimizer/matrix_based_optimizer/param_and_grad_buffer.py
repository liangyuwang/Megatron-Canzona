import torch
from torch.distributed import _coalescing_manager

from megatron.core.utils import is_torch_min_version

import os
from typing import Dict, List, Optional
import logging
import warnings
from contextlib import nullcontext

try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

from ...distributed.param_and_grad_buffer import (
    _ParamAndGradBucketGroup, 
    _ParamAndGradBuffer,
    shard_buffer,
)

from .comm_extension import (
    coalesced_allgather,
    coalesced_reduce_scatter,
    prepare_padded_allgather,
    prepare_padded_reduce_scatter,
)

logger = logging.getLogger(__name__)


class _CombinedWork:
    """Wait on multiple c10d work handles as a single handle."""

    def __init__(self, handles):
        self._handles = [handle for handle in handles if handle is not None]

    def wait(self, *args, **kwargs):
        for handle in self._handles:
            handle.wait(*args, **kwargs)
        return True

    def is_completed(self):
        return all(handle.is_completed() for handle in self._handles)


class _FinalizeCoalescedWork:
    """Finalize padded collectives on their launch streams and chain them to the caller."""

    def __init__(self, handle, plans):
        self._handle = handle
        self._plans = plans
        self._finalized = False
        self._done_events = []

    def _enqueue_finalize(self):
        stream_groups = {}
        for plan in self._plans:
            stream = getattr(plan, "launch_stream", None)
            device = getattr(plan, "device", None)
            if stream is None or device is None or device.type != "cuda":
                plan.finalize()
                continue

            key = (device.index, stream.cuda_stream)
            group = stream_groups.setdefault(
                key, {"device": device, "stream": stream, "plans": []}
            )
            group["plans"].append(plan)

        done_events = []
        for group in stream_groups.values():
            with torch.cuda.stream(group["stream"]):
                for plan in group["plans"]:
                    plan.finalize()
                event = torch.cuda.Event()
                event.record(group["stream"])
            done_events.append((group["device"], event))

        return done_events

    def wait(self, *args, **kwargs):
        if self._handle is not None:
            self._handle.wait(*args, **kwargs)
        if not self._finalized:
            self._done_events = self._enqueue_finalize()
            self._finalized = True
        for device, event in self._done_events:
            torch.cuda.current_stream(device).wait_event(event)
        return True

    def is_completed(self):
        if not self._finalized:
            return False
        if self._handle is not None and not hasattr(self._handle, "is_completed"):
            return False
        handle_completed = self._handle is None or self._handle.is_completed()
        return handle_completed and all(event.query() for _, event in self._done_events)


class _MatrixBasedParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            raise NotImplementedError
        self._buckets_classified = False
        self.even_buckets = []
        self.uneven_buckets = []

    def _classify_buckets(self):
        if not self.ddp_config.use_distributed_optimizer:
            self.even_buckets = self.buckets
            self._buckets_classified = True

        if self._buckets_classified:
            return
        
        has_attr = [hasattr(obj, 'real_gbuf_world_ranges') for obj in self.buckets]
        assert all(has_attr), "'dp buckets' elements must have 'real_gbuf_world_ranges' attribute"
        
        self.even_buckets = []
        self.uneven_buckets = []
        for bucket in self.buckets:
            even_shard = len(set([r.end - r.start for r in bucket.real_gbuf_world_ranges])) == 1
            if even_shard:
                self.even_buckets.append(bucket)
            else:
                self.uneven_buckets.append(bucket)
        
        from megatron.core.utils import log_on_each_pipeline_stage
        buffer_name = self.buckets[0].buffer_name
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f"DP Bucket classification [{buffer_name}]: "
            f"{len(self.even_buckets)} even bucket(s), "
            f"{len(self.uneven_buckets)} uneven bucket(s) "
            f"(total: {len(self.buckets)})"
        )
        
        self._buckets_classified = True

    def _get_uneven_collective_strategy(self) -> str:
        # Strategy choices for uneven DP shards:
        # - "uneven": keep the custom uneven-size collectives and isolate them from
        #   c10d's Python fast-path by running them in a separate coalescing block.
        # - "padded": pad uneven shards up to a common length so both uneven and
        #   even buckets can use the same tensor-collective fast-path.
        # Default to "uneven" because it stays closest to the original data
        # movement pattern and avoids the extra padding/copy overhead.
        strategy = os.environ.get('UNEVEN_COLLECTIVE_STRATEGIES', "uneven")
        if strategy not in {"uneven", "padded"}:
            raise ValueError(
                "UNEVEN_COLLECTIVE_STRATEGIES must be 'uneven' or 'padded', "
                f"got {strategy!r}"
            )
        return strategy

    def _start_param_sync_uneven(self, async_op: bool, data_parallel_rank, data_parallel_group):
        handles = []
        device = self.buckets[0].param_data.device
        with _coalescing_manager(data_parallel_group, device=device, async_ops=async_op) as cm:
            for bucket in self.uneven_buckets:
                total_data_view = [
                    bucket.param_data[r.start - bucket.offset : r.end - bucket.offset]
                    for r in bucket.real_gbuf_world_ranges
                ]
                local_data_view = bucket.param_data[
                    bucket.real_gbuf_world_ranges[data_parallel_rank].start - bucket.offset :
                    bucket.real_gbuf_world_ranges[data_parallel_rank].end - bucket.offset
                ]
                coalesced_allgather(
                    total_data_view, local_data_view, data_parallel_group, async_op
                )
        handles.extend(cm.works)

        with _coalescing_manager(data_parallel_group, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.even_buckets):
                if self.cached_param_buffer_shard_list[idx] is None:
                    self.cached_param_buffer_shard_list[idx] = shard_buffer(
                        bucket.param_data, torch.distributed.get_world_size(data_parallel_group)
                    )
                local_data_view = self.cached_param_buffer_shard_list[idx][data_parallel_rank]
                dist_all_gather_func(
                    bucket.param_data,
                    local_data_view,
                    group=data_parallel_group,
                    async_op=async_op,
                )
        handles.extend(cm.works)
        return _CombinedWork(handles) if async_op and handles else None

    def _start_param_sync_padded(self, async_op: bool, data_parallel_rank, data_parallel_group):
        padded_plans = []
        with _coalescing_manager(data_parallel_group, async_ops=async_op) as cm:
            for bucket in self.uneven_buckets:
                total_data_view = [
                    bucket.param_data[r.start - bucket.offset : r.end - bucket.offset]
                    for r in bucket.real_gbuf_world_ranges
                ]
                local_data_view = bucket.param_data[
                    bucket.real_gbuf_world_ranges[data_parallel_rank].start - bucket.offset :
                    bucket.real_gbuf_world_ranges[data_parallel_rank].end - bucket.offset
                ]
                plan = prepare_padded_allgather(total_data_view, local_data_view)
                plan.run(data_parallel_group, async_op=async_op)
                padded_plans.append(plan)
            for idx, bucket in enumerate(self.even_buckets):
                if self.cached_param_buffer_shard_list[idx] is None:
                    self.cached_param_buffer_shard_list[idx] = shard_buffer(
                        bucket.param_data, torch.distributed.get_world_size(data_parallel_group)
                    )
                local_data_view = self.cached_param_buffer_shard_list[idx][
                    data_parallel_rank
                ]
                dist_all_gather_func(
                    bucket.param_data,
                    local_data_view,
                    group=data_parallel_group,
                    async_op=async_op,
                )
        if async_op:
            return _FinalizeCoalescedWork(cm, padded_plans)
        for plan in padded_plans:
            plan.finalize()
        return None

    def _start_grad_sync_uneven(self, stream_context, async_op: bool, reduce_op, data_parallel_rank, data_parallel_group):
        handles = []
        device = self.buckets[0].grad_data.device
        with stream_context, _coalescing_manager(data_parallel_group, device=device, async_ops=async_op) as cm:
            for bucket in self.uneven_buckets:
                total_data_view = [
                    bucket.grad_data[r.start - bucket.offset : r.end - bucket.offset]
                    for r in bucket.real_gbuf_world_ranges
                ]
                local_data_view = bucket.grad_data[
                    bucket.real_gbuf_world_ranges[data_parallel_rank].start - bucket.offset :
                    bucket.real_gbuf_world_ranges[data_parallel_rank].end - bucket.offset
                ]
                coalesced_reduce_scatter(
                    local_data_view,
                    total_data_view,
                    data_parallel_group,
                    reduce_op,
                    async_op,
                )
        handles.extend(cm.works)

        with stream_context, _coalescing_manager(data_parallel_group, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.even_buckets):
                if self.ddp_config.use_distributed_optimizer:
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, torch.distributed.get_world_size(data_parallel_group)
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][data_parallel_rank]
                    dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=data_parallel_group,
                        async_op=async_op,
                    )
                else:
                    torch.distributed.all_reduce(
                        bucket.grad_data, op=reduce_op, group=data_parallel_group, async_op=async_op
                    )
        handles.extend(cm.works)
        return _CombinedWork(handles) if async_op and handles else None

    def _start_grad_sync_padded(self, stream_context, async_op: bool, reduce_op, data_parallel_rank, data_parallel_group):
        padded_plans = []
        with stream_context, _coalescing_manager(data_parallel_group, async_ops=async_op) as cm:
            for bucket in self.uneven_buckets:
                total_data_view = [
                    bucket.grad_data[r.start - bucket.offset : r.end - bucket.offset]
                    for r in bucket.real_gbuf_world_ranges
                ]
                local_data_view = bucket.grad_data[
                    bucket.real_gbuf_world_ranges[data_parallel_rank].start - bucket.offset :
                    bucket.real_gbuf_world_ranges[data_parallel_rank].end - bucket.offset
                ]
                plan = prepare_padded_reduce_scatter(local_data_view, total_data_view)
                plan.run(data_parallel_group, reduce_op, async_op=async_op)
                padded_plans.append(plan)
            for idx, bucket in enumerate(self.even_buckets):
                if self.ddp_config.use_distributed_optimizer:
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, torch.distributed.get_world_size(data_parallel_group)
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][data_parallel_rank]
                    dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=data_parallel_group,
                        async_op=async_op,
                    )
                else:
                    torch.distributed.all_reduce(
                        bucket.grad_data, op=reduce_op, group=data_parallel_group, async_op=async_op
                    )
        if async_op:
            return _FinalizeCoalescedWork(cm, padded_plans)
        for plan in padded_plans:
            plan.finalize()
        return None

    def start_param_sync(self, force_sync: bool = False):
        """
        Initiates all necessary param all-gathers for this bucket.

        When ddp_config.overlap_param_gather is set to True, dispatches an asynchronous
        communication call (unless force_sync is True). When ddp_config.overlap_param_gather
        is set to False, makes synchronous call.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings if true.
        """
        assert self.ddp_config.use_distributed_optimizer
        data_parallel_rank = torch.distributed.get_rank(group=self.intra_distributed_optimizer_instance_group)

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        
        self._classify_buckets()
        strategy = self._get_uneven_collective_strategy()
        if strategy == "uneven":
            self.param_gather_handle = self._start_param_sync_uneven(async_op, data_parallel_rank, self.intra_distributed_optimizer_instance_group)
        else:
            self.param_gather_handle = self._start_param_sync_padded(async_op, data_parallel_rank, self.intra_distributed_optimizer_instance_group)
        self.param_gather_dispatched = True
    
    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
        """
        Finishes param sync communication operation for this bucket. Dispatches
        next bucket's param sync if available, unless skip_next_bucket_dispatch
        is True.

        When ddp_config.overlap_param_gather is set to True, waits for asynchronous
        communication call to complete (and dispatches one if one is not already
        outstanding). Throws assertion error if ddp_config.overlap_param_gather is set to
        False.

        Args:
            skip_next_bucket_dispatch (bool, optional): if true, dispatch next
                bucket's communication if available.
        """
        assert self.ddp_config.use_distributed_optimizer
        assert self.ddp_config.overlap_param_gather

        # If current bucket's param AG has not been dispatched, dispatch it now (e.g., first
        # AG bucket in first model chunk if ddp_config.align_param_gather is False).
        if not self.param_gather_dispatched:
            self.start_param_sync()

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            # Dispatch next bucket's asynchronous param AG only if it has not been dispatched yet.
            if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                if self.next_param_gather_bucket_group.param_gather_dispatched:
                    warnings.warn(
                        "The next bucket's parameter all-gather operation has already been "
                        "dispatched. This may be caused by a mismatch between the order of "
                        "parameter registration and forward pass execution, which will "
                        "hurt the communication-computation overlap performance."
                    )
                else:
                    self.next_param_gather_bucket_group.start_param_sync()

            # For the mxfp8_param with "reuse_grad_buf_for_mxfp8_param_ag=True",
            # we need to copy the param_data from the shared_param/grad_buffer to param.data
            # after the param all-gather.
            if (
                self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag
                and self.ddp_config.overlap_param_gather
            ):
                for bucket in self.buckets:
                    for param in bucket.params:
                        param_start, param_end = bucket.param_to_index[param]
                        param_slice = bucket.param_data.view(-1)[param_start:param_end]
                        param.data.copy_(param_slice.view(param.data.shape))
                    # All-gathered params are not needed after being copied to param.data.
                    # Zero out the grad buffer (shared with param buffer) for gradient accumulation.
                    bucket.grad_data.zero_()

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # We use the following stream synchronization for the gradient reduction
        # within and across DistOpt instances.

        # Compute Stream: -------------Gradient compute-------------------
        # Comm. Stream:   ------(wait for NCCL)-----(wait for NCCL)-------
        # NCCL Stream:          -------RS------     -------AR------

        # Use async communications only when overlap_grad_reduce is True.
        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )
        if (
            self.ddp_config.num_distributed_optimizer_instances > 1
            and self.ddp_config.overlap_grad_reduce
        ):
            # Assign a communication stream if we have multiple DistOpt instances and we
            # need to overlap communication.
            stream_context = torch.cuda.stream(self.communication_stream)

            # The RS/AR communication stream needs to wait for the default stream
            # to complete its gradient computation before launching the next
            # gradient reduction collective.
            self.communication_stream.wait_stream(torch.cuda.default_stream())
        else:
            stream_context = nullcontext()

        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
        else:
            communication_group = self.data_parallel_group

        data_parallel_rank = torch.distributed.get_rank(group=communication_group)

        self._classify_buckets()
        strategy = self._get_uneven_collective_strategy()
        if strategy == "uneven":
            self.grad_reduce_handle = self._start_grad_sync_uneven(stream_context, async_op, reduce_op, data_parallel_rank, communication_group)
        else:
            self.grad_reduce_handle = self._start_grad_sync_padded(stream_context, async_op, reduce_op, data_parallel_rank, communication_group)

        # With multiple DistOpt instances, we need to all-reduce across instances.
        if (
            self.ddp_config.use_distributed_optimizer
            and self.ddp_config.num_distributed_optimizer_instances > 1
        ):
            assert self.inter_distributed_optimizer_instance_group is not None
            # Create a new coalescing manager for the inter-instance all-reduce.
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    if len(self.uneven_buckets) >= 1:
                        my_range = bucket.real_gbuf_world_ranges[self.intra_distributed_optimizer_instance_rank]
                        local_data_view = bucket.grad_data[
                            my_range.start - bucket.offset : my_range.end - bucket.offset
                        ]
                    else:
                        if self.cached_grad_buffer_shard_list[idx] is None:
                            self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                                bucket.grad_data, self.intra_distributed_optimizer_instance_size
                            )
                        local_data_view = self.cached_grad_buffer_shard_list[idx][
                            self.intra_distributed_optimizer_instance_rank
                        ]

                    torch.distributed.all_reduce(
                        local_data_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )
            
            if async_op:
                self.grad_reduce_handle = cm

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
        communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
        makes synchronous call.
        """
        self.param_gather_dispatched = False
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync()
            return
        # When using multiple DistOpt instances, we don't need to sync here as we launch
        # communications on a separate communication stream.
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            torch.cuda.default_stream().wait_stream(self.communication_stream)
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.params_with_grad)}/{len(self.params)} params have grad available)"
        )
        if self.grad_reduce_handle is not None:
            self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None


def partition_matrix_based_buckets(
    buffers: List[_ParamAndGradBuffer], force_single_bucket_group: bool = False
) -> List[_MatrixBasedParamAndGradBucketGroup]:
    """
    Automatically regroup the buckets of input buffers and return a list of bucket groups.

    In some scenarios, we need to put buckets from different buffers into a group so that their
    communication can be aggregated.

    For example, when there are both fp8 weights and bf16 biases in the model and virtual
    pipeline parallelism is enabled, each model chunk will have an fp8 bucket and a bf16 bucket,
    which doubles the number of communication kernels, and because of the use of
    CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back communications will prevent the
    overlap of communication kernels with computation kernels.

    The grouping strategy is:
    1. If force_single_bucket_group is True, put all buckets across all buffers into a single
       bucket group.
    2. If force_single_bucket_group is False, when there is no fp8 buffer in the input buffers,
       let each bucket group have only one bucket.
    3. If force_single_bucket_group is False, when using fp8 params, merge all non-fp8 buckets
       into the last fp8 bucket group.
       - Since the non-fp8 parameters (typically the biases of various layers) are relatively
         small, they are likely to be grouped into a single non-fp8 bucket.
       - The fp8 buckets start from the end of the model, i.e., the first bucket corresponds to
         the end of the model, while the last bucket corresponds to the beginning.
       - If we combine the non-fp8 bucket with the first fp8 bucket, we cannot initiate the
         reduce-scatter to synchronize gradients after the backward pass at the end of the model
         has completed. This is because we need to wait for the non-fp8 params from the beginning
         layers to obtain their gradients.
       - Combining the non-fp8 bucket with the last fp8 bucket can help avoid this issue.

    Args:
        buffers (list): list of input buffers.
        single_bucket_group_per_buffer (bool, optional): force group all buckets in each buffer
            into a single bucket group.
    """

    if len(buffers) == 0:
        return []

    dtype_to_buffer_map = {}
    for buffer in buffers:
        dtype = buffer.param_dtype
        # Make sure that the param_dtype of any two buffers is different.
        assert dtype not in dtype_to_buffer_map
        dtype_to_buffer_map[dtype] = buffer

    # Case 1: Put all buckets into a single bucket group if force_single_bucket_group is True.
    if force_single_bucket_group:
        buckets = []
        ddp_config = buffers[0].ddp_config
        data_parallel_group = buffers[0].data_parallel_group
        data_parallel_world_size = buffers[0].data_parallel_world_size
        for buffer in buffers:
            assert ddp_config == buffer.ddp_config
            assert data_parallel_group == buffer.data_parallel_group
            assert data_parallel_world_size == buffer.data_parallel_world_size
            buckets.extend(buffer.buckets)

        bucket_group = _MatrixBasedParamAndGradBucketGroup(
            buckets, ddp_config, data_parallel_group, data_parallel_world_size
        )
        return [bucket_group]

    if torch.uint8 not in dtype_to_buffer_map:
        # Case 2: When there is no fp8 buffer in the input buffers, let each bucket group have
        #         only one bucket.
        bucket_groups = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _MatrixBasedParamAndGradBucketGroup(
                        [bucket],
                        buffer.ddp_config,
                        buffer.data_parallel_group,
                        buffer.data_parallel_world_size,
                    )
                )
        return bucket_groups
    else:
        # Case 3: When using fp8 params, merge all non-fp8 buckets into the last fp8 bucket group.
        non_fp8_buckets = []
        for buffer in buffers:
            if buffer.param_dtype != torch.uint8:
                for bucket in buffer.buckets:
                    non_fp8_buckets.append(bucket)

        bucket_groups = []
        fp8_buffer = dtype_to_buffer_map[torch.uint8]
        for bucket in fp8_buffer.buckets:
            if len(bucket_groups) == len(fp8_buffer.buckets) - 1:
                # The last bucket group.
                group_buckets = [bucket] + non_fp8_buckets
            else:
                # The first N-1 bucket groups.
                group_buckets = [bucket]
            bucket_groups.append(
                _MatrixBasedParamAndGradBucketGroup(
                    group_buckets,
                    buffer.ddp_config,
                    buffer.data_parallel_group,
                    buffer.data_parallel_world_size,
                )
            )
        return bucket_groups
