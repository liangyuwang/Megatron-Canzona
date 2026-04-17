import heapq
import logging
from typing import Tuple, Dict, List, Callable, Union
from collections import OrderedDict
import functools, operator
import copy
import random
import math
import os

import torch
import torch.nn as nn
import torch.distributed as dist

from megatron.core.utils import log_on_each_pipeline_stage
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import (
    get_optim_memory_from_param, get_optim_flops_from_param,
)

"""
# Usage
def __init__(self, ...):
    ...
    self.tp_async_param_group_executor = AsyncGroupExecutor(
        # add tp group, rank, world_size
    )

def _single_param_step(self, p, s, ...):
    # define single_param optim step without updating p

def _single_param_update(self, p, tensor_to_update_p, ...):
    # define single_param updating with 'tensor_to_update_p'

def step(self, ...):
    ...
    for group in self.param_groups:
        # update non-tp param first
        ...

        # then update tp param with tp_async_param_group_executor
        self.tp_async_param_group_executor.execute(
            group["params"],
            shapes,
            self._single_param_step,
            self._single_param_update,
            *args,
            balance="single", # ["no", "single", "slot"]
            fused_comm=False,
            **kwargs
        )
"""

class AsyncGroupExecutor:
    def __init__(self,
        group = None,
        rank = None,    # global rank
        world_size = None,
        enable = True
    ):
        self.enable = enable
        self.group = get_tensor_model_parallel_group() if group is None else group
        self.rank = dist.get_rank() if rank is None else rank
        self.world_size = dist.get_world_size(self.group) if world_size is None else world_size
        from megatron.training import get_args
        args = get_args()
        if args.use_tp_balanced_opt:
            self.balance = "global"
        else:
            self.balance = "no"
        if int(os.environ.get('DEBUG_TP_BALANCED_OPT', 0)) == 1:
            self.balance = "single"
        self.fused_comm = args.async_tp_fuse_comm
        self.max_numel_per_slot = int(args.tp_balanced_opt_fuse_space) * 1024 * 1024
        self.tp_cost_name = args.tp_balanced_opt_cost
        self.log_visualization = args.tp_balanced_opt_log_visualization


    def build_param_to_tp_rank_map(self, params, shapes):
        """
        Build and return a mapping from parameters to TP global ranks.

        Args:
            params: List of parameters.
            shapes: List of shapes corresponding to params.

        Returns:
            A dictionary mapping each parameter to its assigned TP global rank.
        """
        param_to_rank_map = {}
        micro_param_groups = self.get_micro_param_groups(self.balance, params, shapes)
        for micro_group in micro_param_groups:
            for global_rank, slot_data in micro_group:
                for p, s in slot_data:
                    param_to_rank_map[p] = global_rank
        return param_to_rank_map

    def execute(
        self,
        param_group,
        shapes,
        param_step_fn,
        param_update_fn,
        *args,
        **kwargs
    ):
        self.param_group=param_group
        if "balanced_micro_tp_groups" not in param_group:
            # we need to rebuild tp balanced_micro_tp_groups before execute, desipte we already compute it when init.
            initialized = False
            param_group["balanced_micro_tp_groups"] = self.get_micro_param_groups(self.balance, param_group['params'], shapes)
        else:
            initialized = True
        if initialized and not param_group["balanced_micro_tp_groups"]:
            return

        log_visualization = self.log_visualization
        if log_visualization and not initialized:
            viz = TPLoadVisualizer(self.world_size, self.rank, self.cost_fn)
            no_balanced_micro_tp_groups = self.get_micro_param_groups("no", param_group['params'], shapes)
            viz.visualize(no_balanced_micro_tp_groups, title="TP No Load Balance Report")
            viz.visualize(param_group["balanced_micro_tp_groups"], title=f"TP Load Balance Report, balance={self.balance}")

        for idx, micro_param_group in enumerate(param_group["balanced_micro_tp_groups"]):
            micro_full_tensors_group = self.gather(
                micro_param_group,)
            micro_full_tensors_to_update_p_group = self.compute(
                micro_param_group,
                micro_full_tensors_group,
                param_step_fn,
                *args,
                group=param_group,
                **kwargs)
            micro_shard_tensors_to_update_p_group = self.scatter(
                micro_param_group,
                micro_full_tensors_to_update_p_group,)
            self.update(
                micro_param_group,
                micro_shard_tensors_to_update_p_group,
                param_update_fn,
                *args,
                group=param_group,
                **kwargs)

    def gather(self, micro_param_group):
        micro_full_tensors_group = [[] for _ in range(self.world_size)]
        if self.fused_comm:
            send_tensors_list = []
            send_split_sizes = []
            ref_dtype = None
            for _, slot in micro_param_group:
                if slot:
                    ref_dtype = slot[0][0].dtype
                    break
            for slot_idx, (host_r, slot) in enumerate(micro_param_group):
                grads_for_host = [p.grad.contiguous().view(-1) for p, s in slot]
                if grads_for_host:
                    fused_send_tensor = torch.cat(grads_for_host)
                else:
                    fused_send_tensor = torch.tensor([], dtype=ref_dtype, device=torch.cuda.current_device())
                send_tensors_list.append(fused_send_tensor.contiguous())
                send_split_sizes.append(fused_send_tensor.numel())
            send_buffer = torch.cat(send_tensors_list).contiguous()
            _, my_param_slot = micro_param_group[get_tensor_model_parallel_rank()]
            my_param_shard_sizes = [get_numel_from_shape(s) for _, s in my_param_slot]
            total_recv_per_rank = sum(my_param_shard_sizes)
            recv_split_sizes = [total_recv_per_rank] * self.world_size
            recv_buffer = torch.empty(sum(recv_split_sizes), dtype=send_buffer.dtype, device=send_buffer.device).contiguous()

            if int(os.environ.get('DEBUG_TP_FUSE_COMM', 0)) == 1:
                work = dist.all_to_all_single(recv_buffer, send_buffer,
                                    input_split_sizes=send_split_sizes,
                                    output_split_sizes=recv_split_sizes,
                                    group=self.group,
                                    async_op=True)
                print(f"[Rank {get_tensor_model_parallel_rank()}] ⏳ Waiting for [Gather] All-to-All completion...", flush=True)
                import time
                start_time = time.time()
                while not work.is_completed():
                    if time.time() - start_time > 10:
                        print(f"[Rank {get_tensor_model_parallel_rank()}] ⚠️ [Gather] All-to-All is taking unusually long (>10s)!", flush=True)
                        start_time = time.time() # reset warning
                    time.sleep(0.01)
                work.wait()
                print(f"[Rank {get_tensor_model_parallel_rank()}] ✅ [Gather] All-to-All Finished!", flush=True)
                dist.barrier()
                torch.cuda.synchronize()
                print(f"[Rank {get_tensor_model_parallel_rank()}] check [Gather] all to all after: {send_buffer.size()}|{sum(send_split_sizes)}|{send_split_sizes}, {recv_buffer.size()}|{sum(recv_split_sizes)}|{recv_split_sizes}", flush=True)
                torch.cuda.synchronize()
                dist.barrier()
            else:
                dist.all_to_all_single(recv_buffer, send_buffer,
                                    input_split_sizes=send_split_sizes,
                                    output_split_sizes=recv_split_sizes,
                                    group=self.group)

            recv_streams = torch.split(recv_buffer, recv_split_sizes)
            stream_offset = 0
            for i, (p, s) in enumerate(my_param_slot):
                shard_numel = my_param_shard_sizes[i]
                shards_to_cat = []
                for src_rank in range(self.world_size):
                    flat_shard = recv_streams[src_rank][stream_offset: stream_offset+shard_numel].contiguous()
                    shards_to_cat.append(flat_shard.view(s))
                full_grad = torch.cat(shards_to_cat, dim=p.partition_dim)
                micro_full_tensors_group[get_tensor_model_parallel_rank()].append(full_grad)
                stream_offset += shard_numel
        else:
            for slot_idx, (host_r, slot) in enumerate(micro_param_group):
                for p, s in slot:
                    g = p.grad.view(s)
                    is_target = host_r == self.rank
                    shard_tensors = [torch.empty_like(g) for _ in range(self.world_size)] if is_target else None
                    dist.gather(g, shard_tensors, dst=host_r, group=self.group, async_op=False)
                    if is_target:
                        micro_full_tensors_group[slot_idx].append(torch.cat(shard_tensors, dim=p.partition_dim))
                    else:
                        micro_full_tensors_group[slot_idx].append(None)
        return micro_full_tensors_group

    def compute(self, micro_param_group, micro_full_tensors_group, param_step_fn, *args, **kwargs):
        micro_full_tensors_to_update_p_group = [[] for _ in range(self.world_size)]
        for (slot_idx, (host_r, slot)), full_tensors_slot in zip(enumerate(micro_param_group), micro_full_tensors_group):
            for (p, s), full_g in zip(slot, full_tensors_slot):
                is_target = host_r == self.rank
                if is_target:
                    full_u = param_step_fn(p, s, *args, g=full_g, **kwargs)
                    micro_full_tensors_to_update_p_group[slot_idx].append(full_u)
                else:
                    micro_full_tensors_to_update_p_group[slot_idx].append(None)
        return micro_full_tensors_to_update_p_group

    def scatter(self, micro_param_group, micro_full_tensors_to_update_p_group):
        micro_shard_tensors_to_update_p_group = [[] for _ in range(self.world_size)]
        if self.fused_comm:
            send_streams = [[] for _ in range(self.world_size)]
            send_split_sizes = []
            ref_dtype = None
            for _, slot in micro_param_group:
                if slot:
                    ref_dtype = slot[0][0].dtype
                    break
            device = torch.cuda.current_device()
            for group_rank_idx, ((host_r, slot), full_updates_slot) in enumerate(zip(micro_param_group, micro_full_tensors_to_update_p_group)):

                if host_r == self.rank:
                    for (p, s), full_u in zip(slot, full_updates_slot):
                        if ref_dtype is None: ref_dtype = full_u.dtype
                        shards = torch.chunk(full_u, self.world_size, dim=p.partition_dim)
                        for target_rank in range(self.world_size):
                            send_streams[target_rank].append(shards[target_rank].contiguous().view(-1))

            flat_send_tensors = []
            for stream in send_streams:
                if stream:
                    t = torch.cat(stream)
                    flat_send_tensors.append(t)
                    send_split_sizes.append(t.numel())
                else:
                    flat_send_tensors.append(torch.tensor([], dtype=ref_dtype, device=device))
                    send_split_sizes.append(0)

            send_buffer = torch.cat(flat_send_tensors).contiguous()

            recv_split_sizes = []
            for src_rank_idx, (host_r, slot) in enumerate(micro_param_group):
                total_recv_from_src = 0
                for p, s in slot:
                    total_recv_from_src += get_numel_from_shape(s)
                recv_split_sizes.append(total_recv_from_src)

            recv_buffer = torch.empty(sum(recv_split_sizes), dtype=send_buffer.dtype, device=device).contiguous()

            if int(os.environ.get('DEBUG_TP_FUSE_COMM', 0)) == 1:
                work = dist.all_to_all_single(recv_buffer, send_buffer,
                    input_split_sizes=send_split_sizes,
                    output_split_sizes=recv_split_sizes,
                    group=self.group,
                    async_op=True)
                print(f"[Rank {get_tensor_model_parallel_rank()}] ⏳ Waiting for [Scatter] All-to-All completion...", flush=True)
                import time
                start_time = time.time()
                while not work.is_completed():
                    if time.time() - start_time > 10:
                        print(f"[Rank {get_tensor_model_parallel_rank()}] ⚠️ [Scatter] All-to-All is taking unusually long (>10s)!", flush=True)
                        start_time = time.time() # reset warning
                    time.sleep(0.01)
                work.wait()
                print(f"[Rank {get_tensor_model_parallel_rank()}] ✅ [Scatter] All-to-All Finished!", flush=True)
                dist.barrier()
                torch.cuda.synchronize()
                print(f"[Rank {get_tensor_model_parallel_rank()}] check [Scatter] all to all after: {send_buffer.size()}|{sum(send_split_sizes)}|{send_split_sizes}, {recv_buffer.size()}|{sum(recv_split_sizes)}|{recv_split_sizes}", flush=True)
                torch.cuda.synchronize()
                dist.barrier()
            else:
                dist.all_to_all_single(recv_buffer, send_buffer,
                    input_split_sizes=send_split_sizes,
                    output_split_sizes=recv_split_sizes,
                    group=self.group)

            recv_streams = torch.split(recv_buffer, recv_split_sizes)

            for src_rank_idx, (host_r, slot) in enumerate(micro_param_group):
                src_stream_flat = recv_streams[src_rank_idx]
                stream_offset = 0

                for p, s in slot:
                    shard_numel = get_numel_from_shape(s)
                    flat_shard = src_stream_flat[stream_offset : stream_offset + shard_numel]
                    micro_shard_tensors_to_update_p_group[src_rank_idx].append(flat_shard.view(s))
                    stream_offset += shard_numel
        else:
            for (slot_idx, (host_r, slot)), full_tensor_to_update_p_slot in zip(enumerate(micro_param_group), micro_full_tensors_to_update_p_group):
                for (p, s), full_tensor_to_update_p in zip(slot, full_tensor_to_update_p_slot):
                    u = torch.empty_like(p.grad.view(s))
                    is_target = host_r == self.rank
                    shard_tensors = list(torch.chunk(full_tensor_to_update_p, chunks=self.world_size, dim=p.partition_dim)) if is_target else None
                    shard_tensors = [shard_tensor.contiguous() for shard_tensor in shard_tensors] if is_target else None
                    dist.scatter(u, shard_tensors, src=host_r, group=self.group, async_op=False)
                    micro_shard_tensors_to_update_p_group[slot_idx].append(u)
        return micro_shard_tensors_to_update_p_group

    def update(self, micro_param_group, micro_shard_tensors_to_update_p_group, param_update_fn, *args, **kwargs):
        for (_, slot), shard_tensor_to_update_p_slot in zip(micro_param_group, micro_shard_tensors_to_update_p_group):
            for (p, s), shard_tensor_to_update_p in zip(slot, shard_tensor_to_update_p_slot):
                param_update_fn(p, shard_tensor_to_update_p, *args, **kwargs)

    def get_micro_groups(self, params, shapes):
        scored_params = []
        for p, s in zip(params, shapes):
            if True:
                scored_params.append((self.cost_fn(p,self.tp_cost_name), p, s))
        if not scored_params:
            return []
        micro_param_groups = []
        for i in range(0, len(scored_params), self.world_size):
            chunk = scored_params[i : i + self.world_size]
            current_group_slots = [[] for _ in range(self.world_size)]
            for rank_offset, (cost, p, s) in enumerate(chunk):
                current_group_slots[rank_offset].append((p, s))
            micro_param_groups.append(current_group_slots)
        return micro_param_groups

    def get_balanced_micro_groups(self, params, shapes):
        scored_params = []
        for p, s in zip(params, shapes):
            if True:
                scored_params.append((self.cost_fn(p,self.tp_cost_name), p, s))
        if not scored_params:
            return []
        scored_params.sort(key=lambda x: x[0], reverse=True)
        balanced_micro_param_groups = []
        for i in range(0, len(scored_params), self.world_size):
            chunk = scored_params[i : i + self.world_size]
            current_group_slots = [[] for _ in range(self.world_size)]
            for rank_offset, (cost, p, s) in enumerate(chunk):
                current_group_slots[rank_offset].append((p, s))
            balanced_micro_param_groups.append(current_group_slots)
        return balanced_micro_param_groups

    def get_balanced_slot_micro_group(self, params, shapes, max_numel_per_slot=400*1024*1024):
        scored_params = []
        for p, s in zip(params, shapes):
            if True:
                scored_params.append((self.cost_fn(p,self.tp_cost_name), p.numel(), p, s)) # (target cost, comm cost, p, s)
        if not scored_params:
            return []
        scored_params.sort(key=lambda x: x[0], reverse=True)

        max_single_comm_size = max(item[1] for item in scored_params)
        if max_numel_per_slot < max_single_comm_size:
            max_numel_per_slot = max_single_comm_size

        balanced_micro_param_groups = []
        current_slots = [[] for _ in range(self.world_size)]
        slot_target_loads = [0.0] * self.world_size
        slot_comm_loads = [0] * self.world_size
        target_load_heap = [(0.0, i) for i in range(self.world_size)]
        heapq.heapify(target_load_heap)

        for target_cost, comm_cost, p, s in scored_params:
            min_target_load, slot_idx = heapq.heappop(target_load_heap)
            current_slots[slot_idx].append((p, s))

            new_target_load = min_target_load + target_cost
            slot_target_loads[slot_idx] = new_target_load
            heapq.heappush(target_load_heap, (new_target_load, slot_idx))
            slot_comm_loads[slot_idx] += comm_cost

            if max(slot_target_loads) >= max_numel_per_slot:
                balanced_micro_param_groups.append(current_slots)

                current_slots = [[] for _ in range(self.world_size)]
                slot_target_loads = [0.0] * self.world_size
                slot_comm_loads = [0] * self.world_size
                target_load_heap = [(0.0, i) for i in range(self.world_size)]
                heapq.heapify(target_load_heap)

        if any(len(slot) > 0 for slot in current_slots):
            balanced_micro_param_groups.append(current_slots)

        return balanced_micro_param_groups

    def get_globally_optimized_micro_groups(self, params, shapes, max_numel_per_slot=400*1024*1024):
        """
        Args:
            params: list of flattened parameters
            shapes: list of original shapes
            max_numel_per_slot: soft/hard upper bound for slot capacity
            cost_fn: optional function, signature cost_fn(param_tensor) -> int/float.
                    If None, defaults to param.numel().
        """
        def partition_solve(items):
            sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

            slot_loads = [0] * self.world_size
            allocation = [[] for _ in range(self.world_size)]

            pq = [(0, i) for i in range(self.world_size)]

            for item in sorted_items:
                cost, _, p, s = item

                min_load, slot_idx = heapq.heappop(pq)

                new_load = min_load + cost
                slot_loads[slot_idx] = new_load
                allocation[slot_idx].append((p, s))

                heapq.heappush(pq, (new_load, slot_idx))

            max_load = max(slot_loads)
            return allocation, max_load

        meta_items = []

        for i, (p, s) in enumerate(zip(params, shapes)):
            if True:
                original_view = p.view(s)
                cost = self.cost_fn(original_view, self.tp_cost_name)

                if cost > max_numel_per_slot:
                    raise ValueError(
                        f"Parameter at index {i} has cost {cost}, which exceeds "
                        f"max_numel_per_slot {max_numel_per_slot}."
                    )

                meta_items.append((cost, i, p, s))

        if not meta_items:
            return []

        meta_items.sort(key=lambda x: (x[0], x[1]), reverse=True)

        optimized_micro_groups = []

        current_candidate_items = []
        current_candidate_cost_sum = 0

        idx = 0
        total_items = len(meta_items)

        while idx < total_items:
            item = meta_items[idx]
            item_cost = item[0]

            current_candidate_items.append(item)
            current_candidate_cost_sum += item_cost

            avg_load = current_candidate_cost_sum / self.world_size
            is_overflow = False

            if avg_load > max_numel_per_slot:
                is_overflow = True
            else:
                allocation, max_load = partition_solve(current_candidate_items)
                if max_load > max_numel_per_slot:
                    is_overflow = True

            if is_overflow:
                current_candidate_items.pop()
                current_candidate_cost_sum -= item_cost

                if not current_candidate_items:
                    current_candidate_items.append(item)
                    idx += 1
                    alloc, _ = partition_solve(current_candidate_items)
                    optimized_micro_groups.append(alloc)
                    current_candidate_items = []
                    current_candidate_cost_sum = 0
                    continue

                valid_allocation, _ = partition_solve(current_candidate_items)
                optimized_micro_groups.append(valid_allocation)

                current_candidate_items = []
                current_candidate_cost_sum = 0

            else:
                idx += 1

        if current_candidate_items:
            final_allocation, _ = partition_solve(current_candidate_items)
            optimized_micro_groups.append(final_allocation)

        return optimized_micro_groups

    def get_micro_param_groups(self, balance, *args, **kwargs):
        match balance:
            case "no":
                micro_param_groups = self.get_micro_groups(*args, **kwargs)
            case "single":
                micro_param_groups = self.get_balanced_micro_groups(*args, **kwargs)
            case "slot":
                micro_param_groups = self.get_balanced_slot_micro_group(*args, max_numel_per_slot=self.max_numel_per_slot, **kwargs)
            case "global":
                micro_param_groups = self.get_globally_optimized_micro_groups(*args, max_numel_per_slot=self.max_numel_per_slot, **kwargs)
            case _:
                raise NotImplementedError
        if not micro_param_groups:
            return []
        final_micro_param_groups = []
        for slots_data in micro_param_groups:
            group_global_ranks = [
                dist.get_global_rank(self.group, i)
                for i in range(self.world_size)
            ]
            weighted_group = list(zip(group_global_ranks, slots_data))
            final_micro_param_groups.append(weighted_group)
        return final_micro_param_groups

    def cost_fn(self, p, cost_name):
        if cost_name == "numel":
            return p.numel()
        elif cost_name == "flops":
            return get_optim_flops_from_param(p)
        else:
            raise ValueError


class SyncGroupExecutor:
    """
    For debug and experiments
    """
    def __init__(self,
        group = None,
        rank = None,
        world_size = None,
        enable = True
    ):
        self.enable = enable
        self.group = get_tensor_model_parallel_group() if group is None else group
        self.rank = get_tensor_model_parallel_rank() if rank is None else rank
        self.world_size = get_tensor_model_parallel_world_size() if world_size is None else world_size

    def execute(
        self,
        param_group,
        shapes,
        param_step_fn,
        param_update_fn,
        *args,
        **kwargs
    ):
        for p, shape in zip(param_group['params'], shapes):
            if not (is_group_tensor_parallel(param_group) and p.grad is not None):
                continue
            local_grad = p.grad.view(shape)
            gathered_shards = [torch.empty_like(local_grad) for _ in range(self.world_size)]
            dist.all_gather(gathered_shards, local_grad, group=self.group)
            full_grad = torch.cat(gathered_shards, dim=p.partition_dim)

            full_update = param_step_fn(p, shape, *args, group=param_group, g=full_grad, **kwargs)

            shards = torch.chunk(full_update, self.world_size, dim=p.partition_dim)
            local_update_shard = shards[self.rank].contiguous()
            param_update_fn(p, local_update_shard, *args, group=param_group, **kwargs)


def is_group_tensor_parallel(group):
    return group['is_tensor_parallel'] and get_tensor_model_parallel_world_size() > 1


def get_numel_from_shape(shape):
    return functools.reduce(operator.mul, shape, 1)



# =================================
# ------- visual utility ---------
# =================================

logger = logging.getLogger(__name__)

def _log(msg: str) -> None:
    log_on_each_pipeline_stage(logger, logging.INFO, msg)

class TPLoadVisualizer:
    def __init__(self, world_size, rank=0, cost_fn=None):
        self.world_size = world_size
        self.rank = rank
        # ANSI colors for terminal output
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'warning': '\033[93m',
            'fail': '\033[91m',
            'end': '\033[0m',
            'bold': '\033[1m'
        }
        self.bar_char = "█"
        self.empty_char = "░"
        self.cost_fn = cost_fn

    def _human_readable_size(self, numel, dtype_bytes=4):
        """Converts numel to a human-readable size string (B, KB, MB, GB)."""
        # Note: The original comment "fp32 (2 bytes)" was a typo. FP32 is 4 bytes.
        size_bytes = numel * dtype_bytes
        if size_bytes <= 0: return "0 B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    def _human_readable_unit(self, value, unit=""):
        """Generic function to format large numbers with metric prefixes (K, M, B)."""
        if value == 0: return f"0 {unit}"
        if abs(value) < 1000: return f"{int(value)} {unit}"
        unit_prefixes = ('', 'K', 'M', 'B', 'T', 'P') # Using B for Billion, T for Trillion
        i = int(math.floor(math.log(abs(value), 1000))) if abs(value) > 0 else 0
        i = min(i, len(unit_prefixes) - 1)
        p = math.pow(1000, i)
        s = round(value / p, 2)
        return f"{s} {unit_prefixes[i]}{unit}"

    def _human_readable(self, cost_name, *args):
        if cost_name == "numel":
            return self._human_readable_size(*args)
        elif cost_name == "flops":
            return self._human_readable_unit(*args)
        else:
            raise ValueError

    def _get_color_by_load(self, load, max_load, min_load):
        if max_load == min_load:
            return self.colors['green']
        ratio = (load - min_load) / (max_load - min_load + 1e-6)
        if ratio > 0.8: return self.colors['fail']
        if ratio < 0.2: return self.colors['blue']
        return self.colors['green']

    def visualize_cost(self, micro_param_groups, cost_name, title="TP Load Balance Report"):
        """
        Main function to visualize the groups.
        Args:
            micro_param_groups: The output from AsyncGroupExecutor.get_micro_param_groups
        """
        cost_fn_ = lambda p: self.cost_fn(p, cost_name)

        _log(f"\n=== {title} ===")
        _log(f"Total Micro Groups: {len(micro_param_groups)}\n")

        total_imbalance_params = 0
        total_params = 0

        for g_idx, group in enumerate(micro_param_groups):
            # group structure: [(global_rank, slot_params), ...]
            # slot_params: [(p, s), ...]

            # 1. Calculate loads for this micro group
            rank_loads = []
            for global_r, slot_params in group:
                load = sum(cost_fn_(p.view(s)) for p, s in slot_params)
                rank_loads.append(load)

            max_load = max(rank_loads)
            min_load = min(rank_loads)
            avg_load = sum(rank_loads) / len(rank_loads) if rank_loads else 0
            imbalance_ratio = max_load / avg_load if avg_load > 0 else 1.0

            total_params += sum(rank_loads)
            total_imbalance_params += (sum(rank_loads) - min_load * len(rank_loads))

            # 2. Print Group Header
            _log(f"Micro Group [{g_idx}] "
                f"(Max Imbalance: {imbalance_ratio:.4f}x | "
                f"Spread: {self._human_readable(cost_name, max_load - min_load)})")

            # 3. Print Bars for each Rank
            max_bar_width = 40
            for r_idx, load in enumerate(rank_loads):
                # Calculate bar width
                bar_len = int((load / max_load) * max_bar_width) if max_load > 0 else 0
                bar_str = self.bar_char * bar_len + self.empty_char * (max_bar_width - bar_len)

                size_str = self._human_readable(cost_name, load)
                _log(f"  Rank {r_idx}: [{bar_str}] {size_str}")
            _log("-" * 60)

        _log("Summary:")
        _log(f"  Total Processed: {self._human_readable(cost_name, total_params)}")
        efficiency = (1.0 - (total_imbalance_params / total_params)) * 100 if total_params > 0 else 100
        _log(f"  Approx. Computational Efficiency: {efficiency:.4f}%")
        _log("=" * 60 + "\n")

    def visualize(self, micro_param_groups, title="TP Load Balance Report"):
        self.visualize_cost(micro_param_groups, cost_name="numel", title=f"{title} (Memory View)")
        self.visualize_cost(micro_param_groups, cost_name="flops", title=f"{title} (FLOPs View)")