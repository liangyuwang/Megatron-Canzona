import os
import gc
import itertools
import logging
import warnings
import math
from dataclasses import replace
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional

from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.optimizer.optimizer import (
    MixedPrecisionOptimizer,
    _multi_tensor_copy_this_to_that,
    _zero_grad_group_helper,
)
from megatron.core.utils import log_single_rank
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

from ... import tensor_parallel
from ...config_logger import has_config_logger_enabled, log_config_to_disk
from ...dist_checkpointing import ShardedTensor
from ...dist_checkpointing.dict_utils import nested_values
from ...dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedStateDict,
    ShardedTensorFactory,
)
from ...dist_checkpointing.utils import extract_sharded_tensors_and_factories
from ...distributed.param_and_grad_buffer import _ParamAndGradBuffer
from ...transformer.module import MegatronModule
from ...fp8_utils import is_float8tensor
from ..grad_scaler import MegatronGradScaler
from ..optimizer import param_group_identifier_keys
from ..optimizer_config import OptimizerConfig
from ..distrib_optimizer import (
    USING_TE_OPTIMIZER,
    USING_APEX_OPTIMIZER,
    KEEP_VARS_HINT,
    DistributedOptimizer, 
    Range
)
from ..distrib_optimizer import HAVE_APEX_OR_TE, Adam

from .param_and_grad_buffer import partition_matrix_based_buckets
from .optimizers import Muon
from .optimizers import SOAP
from .utils import is_matrix_based_optim, is_matrix_based_optim_group

try:
    # This will be used when "--fp8-param-gather" is enabled.
    # When BF16/FP16 parameters don't exist, we need to cast the FP32 main parameters to
    # FP8 directly in the optimizer.
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
except:
    pass

logger = getLogger(__name__)


class DistMatrixBasedOptimizer(DistributedOptimizer, MixedPrecisionOptimizer):

    @classmethod
    def _build_model_gbuf_param_range_map_whole_tensor(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        virtual_gbuf_world_all_ranges: Range,
        bucket_offset: int,
        cur_dp_rank: int,
        build_local = True,
    ):
        param_range_map = {}
        real_gbuf_world_ranges = []
        if not build_local:
            bucket_dp_slices_pos = []
        # for uneven sharding, we need to record the real range of each dp rank
        for dp_rank, virtual_gbuf_world_range in enumerate(virtual_gbuf_world_all_ranges):
            real_gbuf_world_start = None
            real_gbuf_world_end = None
            for param, param_world_indexes in param_world_index_map.items():
                param_world_start, param_world_end, _ = param_world_indexes
                if virtual_gbuf_world_range.start <= param_world_start < virtual_gbuf_world_range.end:
                    if real_gbuf_world_start is None:
                        real_gbuf_world_start = param_world_start
                    else:
                        assert real_gbuf_world_start < param_world_start, "we assume params are ordered"
                    if real_gbuf_world_end is None or param_world_end > real_gbuf_world_end:
                        real_gbuf_world_end = param_world_end
                    param_local_start = param_world_start - real_gbuf_world_start
                    param_local_end = param_world_end - real_gbuf_world_start

                    param_local_range = Range(param_local_start, param_local_end)
                    param_world_range = Range(param_world_start, param_world_end)

                    param_world_range_in_bucket = Range(
                        param_world_range.start - bucket_offset,
                        param_world_range.end - bucket_offset
                    )

                    sub_param_range = Range(0, param_world_end - param_world_start)
                    if build_local:
                        if dp_rank == cur_dp_rank:
                            # only record the param range map for current dp rank
                            param_range_map[param] = {
                                "gbuf_world": param_world_range, # param position in whole buffer
                                "gbuf_world_in_bucket": param_world_range_in_bucket, # param position in current bucket
                                "gbuf_local": param_local_range, # param position in local shard
                                "param": sub_param_range, # shard position in param
                            }
                    else:
                        param_range_map[param] = {
                            "gbuf_world": param_world_range, # param position in whole buffer
                            "gbuf_world_in_bucket": param_world_range_in_bucket, # param position in current bucket
                            "gbuf_local": param_local_range, # param position in local shard
                            "param": sub_param_range, # shard position in param
                        }
            if real_gbuf_world_start is None:
                real_gbuf_world_range = Range(virtual_gbuf_world_range.start, virtual_gbuf_world_range.start) # empty range
            else:
                real_gbuf_world_range = Range(real_gbuf_world_start, real_gbuf_world_end)
            real_gbuf_world_ranges.append(real_gbuf_world_range)

            if not build_local:
                if real_gbuf_world_end is None:
                    if dp_rank == 0:
                        bucket_dp_slices_pos.append(virtual_gbuf_world_range.start - bucket_offset)
                    else:
                        bucket_dp_slices_pos.append(bucket_dp_slices_pos[-1])
                else:
                    bucket_dp_slices_pos.append(real_gbuf_world_end - bucket_offset)

        if build_local:
            return param_range_map, real_gbuf_world_ranges
        else:
            param_to_in_bucket_range_map = {
                param: (p_indices[0] - bucket_offset, p_indices[1] - 1 - bucket_offset)
                for param, p_indices in param_world_index_map.items()
            }
            # check
            for param, p_range in param_to_in_bucket_range_map.items():
                assert param.numel() - 1 == p_range[1] - p_range[0], \
                    f"'param.numel() - 1': {param.numel() - 1} should be equal to 'p_range[1] - p_range[0]': {p_range[1] - p_range[0]}"
            return param_range_map, real_gbuf_world_ranges, param_to_in_bucket_range_map, [0]+bucket_dp_slices_pos

    @classmethod
    def _build_dp_load_balanced_model_gbuf_param_range_map_whole_tensor(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        bucket_dp_slices_pos: list,
        bucket_offset: int,
        cur_dp_rank: int,
    ):
        real_gbuf_world_ranges = []
        dp_size = len(bucket_dp_slices_pos) - 1
        for i in range(dp_size):
            start = bucket_dp_slices_pos[i] + bucket_offset
            end = bucket_dp_slices_pos[i+1] + bucket_offset
            real_gbuf_world_ranges.append(Range(start, end))

        param_range_map = {}
        # for uneven sharding, we need to record the real range of each dp rank
        for dp_rank in range(len(bucket_dp_slices_pos) - 1):
            dp_slice_world_range = real_gbuf_world_ranges[dp_rank]
            dp_slice_world_start = dp_slice_world_range.start
            dp_slice_world_end = dp_slice_world_range.end
            for param, param_world_indexes in param_world_index_map.items():
                param_world_start, param_world_end, _ = param_world_indexes
                if dp_slice_world_start <= param_world_start < dp_slice_world_end:
                    assert param_world_end <= dp_slice_world_end, \
                        f"Parameter {param.shape} with world range [{param_world_start}, {param_world_end}) " \
                        f"is split by pre-computed slice for DP rank {dp_rank} with world range " \
                        f"[{dp_slice_world_start}, {dp_slice_world_end})."
                    param_local_start = param_world_start - dp_slice_world_start
                    param_local_end = param_world_end - dp_slice_world_start

                    param_local_range = Range(param_local_start, param_local_end)
                    param_world_range = Range(param_world_start, param_world_end)

                    param_world_range_in_bucket = Range(
                        param_world_range.start - bucket_offset,
                        param_world_range.end - bucket_offset
                    )

                    sub_param_range = Range(0, param_world_end - param_world_start)
                    if dp_rank == cur_dp_rank:
                        # only record the param range map for current dp rank
                        param_range_map[param] = {
                            "gbuf_world": param_world_range, # param position in whole buffer
                            "gbuf_world_in_bucket": param_world_range_in_bucket, # param position in current bucket
                            "gbuf_local": param_local_range, # param position in local shard
                            "param": sub_param_range, # shard position in param
                        }
        return param_range_map, real_gbuf_world_ranges

    @classmethod
    def _build_model_gbuf_range(cls,
                                param_and_grad_buffer: _ParamAndGradBuffer,
                                bucket_index: int,
                                use_matrix_based_optim=False,
                                buffer_dp_slices_pos: list=None,
                                dp_balance=False):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = param_and_grad_buffer.data_parallel_group.rank()
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel()
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"

        if not (use_matrix_based_optim and dp_balance and buffer_dp_slices_pos is not None):
            max_gbuf_range_size = gbuf_size // data_parallel_world_size
            # All world ranges (i.e., across all data parallel ranks).
            gbuf_world_all_ranges = []
            for r in range(data_parallel_world_size):
                # Compute start of chunk in this bucket.
                gbuf_world_start = r * max_gbuf_range_size
                gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
                # Add bucket's offset in grad buffer.
                gbuf_world_range = Range(
                    gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
                )
                gbuf_world_all_ranges.append(gbuf_world_range)

        # Get each param's ranges.
        if use_matrix_based_optim:
            if dp_balance:
                if buffer_dp_slices_pos is not None:
                    param_range_map, real_bucket_ranges = cls._build_dp_load_balanced_model_gbuf_param_range_map_whole_tensor(
                        param_and_grad_buffer.param_index_map, buffer_dp_slices_pos[bucket_index], bucket.offset, data_parallel_rank
                    )
                    bucket.real_gbuf_world_ranges = real_bucket_ranges
                else:
                    param_range_map, _, param_to_in_bucket_range_map, bucket_dp_slices_pos = cls._build_model_gbuf_param_range_map_whole_tensor(
                        param_and_grad_buffer.param_index_map, gbuf_world_all_ranges, bucket.offset, data_parallel_rank, build_local=False
                    )
            else:
                param_range_map, real_bucket_ranges = cls._build_model_gbuf_param_range_map_whole_tensor(
                    param_and_grad_buffer.param_index_map, gbuf_world_all_ranges, bucket.offset, data_parallel_rank
                )
                bucket.real_gbuf_world_ranges = real_bucket_ranges
        else:
             # Local DP's ranges.
            gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]
            param_range_map = cls._build_model_gbuf_param_range_map(
                param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
            )

        # Group into dict.
        data = {"param_map": param_range_map}

        if use_matrix_based_optim and dp_balance and buffer_dp_slices_pos is None:
            return data, param_to_in_bucket_range_map, bucket_dp_slices_pos
        else:
            return data

    @classmethod
    def _build_gbuf_range_map(cls,
                              param_and_grad_buffer: _ParamAndGradBuffer,
                              use_matrix_based_optim=False,
                              dp_balance=False):
        """
        Build mapping between params and their grad buffers. These mappings are
        partitioned according to data type.

        Iterate through all buckets of grad buffer to construct param ranges
        that this rank "owns" (the dp_rank'th shard of each bucket, where each
        shard is 1/dp_world_size of the bucket).

        Load-balance dev logic:
            if use_matrix_based_optim:
                if dp_balance:
                    .load_balanced_dp_buffer.build_dp_load_balanced_dist_opt_buffer_slices_pos
                        -> cls._build_model_gbuf_range(use_matrix_based_optim=True, buffer_dp_slices_pos=None, dp_balance=True)
                            -> cls._build_model_gbuf_param_range_map_whole_tensor(build_local=False)
                        -> greedy_lpt_with_ranges
                        return -> (buffer_dp_slices_pos)
                    cls._build_model_gbuf_range(use_matrix_based_optim=True, buffer_dp_slices_pos=buffer_dp_slices_pos, dp_balance=True)
                        -> cls._build_dp_load_balanced_model_gbuf_param_range_map_whole_tensor(bucket_dp_slices_pos=buffer_dp_slices_pos[bucket_index])
                        return -> (gbuf_range_map)
                else:
                    cls._build_model_gbuf_range(use_matrix_based_optim=True, buffer_dp_slices_pos=None, dp_balance=False)
                        -> cls._build_model_gbuf_param_range_map_whole_tensor(build_local=True)
                        return -> (gbuf_range_map)
            else:
                cls._build_model_gbuf_range(use_matrix_based_optim=False, buffer_dp_slices_pos=None, dp_balance=False)
                    -> cls._build_model_gbuf_param_range_map
                    return -> (gbuf_range_map)

        Args:
            param_and_grad_buffer (_ParamAndGradBuffer): buffer to build mapping for.
            use_matrix_based_optim (bool): enable matrix-based optimizer
            dp_balance (bool): enable dp load-balance
        """
        buffer_dp_slices_pos = None
        if dp_balance and use_matrix_based_optim:
            from .load_balanced_dp_buffer import build_dp_load_balanced_dist_opt_buffer_slices_pos
            buffer_dp_slices_pos = build_dp_load_balanced_dist_opt_buffer_slices_pos(
                param_and_grad_buffer,
                dp_size=param_and_grad_buffer.data_parallel_group.size(),
            )
        return {
            (param_and_grad_buffer.param_dtype, param_and_grad_buffer.grad_dtype): [
                cls._build_model_gbuf_range(
                    param_and_grad_buffer,
                    bucket_index,
                    use_matrix_based_optim=use_matrix_based_optim,
                    buffer_dp_slices_pos=buffer_dp_slices_pos,
                    dp_balance=dp_balance)
                for bucket_index in range(len(param_and_grad_buffer.buckets))
            ]
        }

    @classmethod
    def _build_model_and_main_param_groups(
        cls,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
        optim_splitter,
        config: OptimizerConfig,
    ):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Allocate (or slice) each group's param shard.
        for group_range in opt_group_ranges:

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)
            group_range["model_to_main_pairs"] = []
            group_range["orig_group"]["params"] = []
            orig_group_use_matrix_based_optimizer = is_matrix_based_optim_group(group_range['orig_group'])
            if orig_group_use_matrix_based_optimizer:
                group_range["orig_group"]["origin_shape"] = []
            for model_param in group_range["params"]:

                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]
                true_attrs=[]
                if orig_group_use_matrix_based_optimizer:
                    group_range["orig_group"]["origin_shape"].append(model_param.shape)

                    true_attrs = optim_splitter.get_split_param_methods(model_param) if optim_splitter is not None else []
                    if true_attrs:
                        assert len(true_attrs) == 1, f"Only one of {optim_splitter.get_attrs()} can be set for a param, got {true_attrs}"
                        attr_name = true_attrs[0]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                    # Generate sharded model param.
                    if is_float8tensor(model_param) and config.fp8_recipe != "delayed":
                        # MXFP8Tensor and BlockwiseQTensor don't support view(-1)
                        shard_model_param = None
                    else:
                        shard_model_param = model_param.detach().view(-1)[
                            param_range.start : param_range.end
                        ]
                        tensor_parallel.copy_tensor_model_parallel_attributes(
                            shard_model_param, model_param
                        )
                        if hasattr(model_param, 'shared'):
                            shard_model_param.shared = model_param.shared

                    # Generate main param.
                    if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                        # If we use FP8 params to initialize FP32 main params (compared to using the
                        # bf16/fp16 params to initialize the main params), there will be a loss of
                        # precision at the beginning of training (this problem will not occur if the
                        # training is long enough or if the main params are loaded from a checkpoint).
                        if is_float8tensor(model_param) and hasattr(
                            model_param, 'get_high_precision_init_val'
                        ):
                            if hasattr(model_param, 'get_high_precision_init_val'):
                                shard_main_param = (
                                    model_param.get_high_precision_init_val()
                                    .view(-1)[param_range.start : param_range.end]
                                    .clone()
                                    .to(shard_model_param.device)
                                    .float()
                                )
                                model_param.clear_high_precision_init_val()
                            else:
                                shard_main_param = model_param.float().view(-1)[
                                    param_range.start : param_range.end
                                ]
                        else:
                            shard_main_param = shard_model_param.clone().float()

                        tensor_parallel.copy_tensor_model_parallel_attributes(
                            shard_main_param, model_param
                        )
                        if hasattr(model_param, 'shared'):
                            shard_main_param.shared = model_param.shared
                    else:
                        # When using precision-aware optimizer, main params are held by FusedAdam.
                        shard_main_param = None

                    # Store handle to main_param.
                    model_param.main_param = shard_main_param
                    model_param.main_param_sharded = True

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)
                    group_range["orig_group"]["params"].append(shard_main_param)
                    group_range["model_to_main_pairs"].append((model_param, shard_main_param))
                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                    group_range["orig_group"]["params"].append(shard_model_param)
                    # For fp32, main param is the shard_model_param
                    group_range["model_to_main_pairs"].append((model_param, shard_model_param))

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )
                if true_attrs:
                    setattr(group_range["orig_group"]["params"][-1], attr_name, True)

            # Update optimizer's params.
            if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_fp32_from_float16_params_this_group,
                ]
            else:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_float16_params_this_group,
                ]
            
        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        model_chunks: List[MegatronModule],
        per_model_buffers: Dict[int, List[_ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
        data_parallel_group_idx: int,
        distributed_optimizer_instance_id: int,
    ):
        """
        Distributed optimizer, for all data types (fp16, bf16, and fp32).

        The steps in this method create the core mapping between param and grad buffers,
        parameters, and parameter shard ranges, that is needed for converting between model
        param indexes and main parameter shard indexes. This method also updates the optimizer
        parameter groups with the newly created shards.

        Args:
            optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
            config (OptimizerConfig): configuration object for optimizer.
            grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
                this can be None. This case happens when `bf16 = True` and we don't
                use any loss scale. Note that for `bf16 = True`, we can have
                a constant gradient scaler. Also for `bf16 = False`, we
                always require a grad scaler.
            init_state_fn (Callable, optional): function to initialize state in the optimizer.
            model_chunks (List[MegatronModule]): list of model chunks.
            per_model_buffers (Dict[int, List[_ParamAndGradBuffer]]): the implementation of the
                distributed optimizer is centered on using a contiguous buffer for
                communicating grads & params between the model state and the optimizer state.
                You can find a more detailed description in
                https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md.
            data_parallel_group (torch.distributed.ProcessGroup): data-parallel group to use to
                all-gather params after optimizer.step().
            data_parallel_group_gloo (torch.distributed.ProcessGroup): gloo data-parallel group
                (used in checkpoint loading and saving).
            data_parallel_group_idx (int): index in data-parallel group (used by
                distributed checkpointing logic).
            distributed_optimizer_instance_id (int): index of the Distributed Optimizer instance.
        """

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        MixedPrecisionOptimizer.__init__(self, optimizer, config, grad_scaler, init_state_fn)
        self.model_chunks = model_chunks
        self.ddp_config = self.model_chunks[0].ddp_config
        for model_chunk in self.model_chunks:
            assert self.ddp_config == model_chunk.ddp_config
        self.distributed_optimizer_instance_id = distributed_optimizer_instance_id
        
        assert not isinstance(optimizer, HybridDeviceOptimizer), (
            "HybridDeviceOptimizer currently is not supported"
        )
        self.is_stub_optimizer = False
        assert distributed_optimizer_instance_id == 0, (
            "Multiple DistributedOptimizer instances are not supported. "
            f"Got distributed_optimizer_instance_id={distributed_optimizer_instance_id}."
        )
        assert not self.ddp_config.use_megatron_fsdp, "Megatron FSDP is currently not supported"

        if isinstance(optimizer, Adam):
            self.use_optimizer = 'adam'
        elif isinstance(optimizer, Muon):
            self.use_optimizer = 'muon'
        elif isinstance(optimizer, SOAP):
            self.use_optimizer = 'soap'
        else:
            warnings.warn("Only Adam currently supported, due to checkpointing requirements.")

        if self.config.split_matrix_based_optimizer_params and is_matrix_based_optim(self.use_optimizer):
            self.optim_splitter = optimizer.grad_and_state_splitter
        else:
            self.optim_splitter = None

        # Model grad buffer ranges.
        assert per_model_buffers is not None, "per_model_buffers must be provided"
        self.buffers = list(itertools.chain(*per_model_buffers.values()))
        self.per_model_buffers = per_model_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo
        self.data_parallel_group_idx = data_parallel_group_idx

        self.gbuf_idx_to_model_idx_map = {}
        gbuf_idx = 0
        for model_idx, buffers in self.per_model_buffers.items():
            for _ in buffers:
                self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
                gbuf_idx += 1

        self.per_model_bucket_groups = {}
        for model_idx, buffers in self.per_model_buffers.items():
            self.per_model_bucket_groups[model_idx] = partition_matrix_based_buckets(buffers)

        self.gbuf_ranges = []
        self.per_bucket_numel = []
        self.per_bucket_numel_unpadded = []
        for buffer in self.buffers:

            self.per_bucket_numel.append(
                {
                    (buffer.param_dtype, buffer.grad_dtype): [
                        bucket.grad_data.numel() for bucket in buffer.buckets
                    ]
                }
            )
            self.per_bucket_numel_unpadded.append(
                {
                    (buffer.param_dtype, buffer.grad_dtype): [
                        bucket.numel_unpadded for bucket in buffer.buckets
                    ]
                }
            )
            self.gbuf_ranges.append(self._build_gbuf_range_map(
                buffer,
                is_matrix_based_optim(),
                self.config.use_dp_balanced_opt))
        self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

        # Add main_param field to each parameter. We will use this fp32 copy to compute
        # the param norm.
        # For parameters with optimizer state on this rank, None will be overwritten by
        # the corresponding sharded main_param tensor.
        for param_group in self.optimizer.param_groups:
            # For all the parameters in this group.
            for param in param_group['params']:
                if param.requires_grad:
                    # fp32 copy only needed for 16-bit parameters.
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                        param.main_param = None
                        param.main_param_sharded = True

        # Optimizer ranges.
        (self.model_param_group_index_map, self.opt_group_ranges) = (
            self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)
        )

        # Allocate main param shards.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
        ) = self._build_model_and_main_param_groups(
            self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges, self.optim_splitter, config
        )

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer = HybridDeviceOptimizer(
                params=[g["orig_group"] for g in self.opt_group_ranges], **self.optimizer.defaults
            )
        else:
            self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
            self.optimizer.load_state_dict(self.optimizer.state_dict())
        
        # Build param_to_tp_rank_map for async TP state allocation
        from megatron.training import get_args
        args = get_args()
        if args.use_tp_async_opt:
            self._build_param_to_tp_rank_map()

    def _build_param_to_tp_rank_map(self):
        """
        Build a mapping from model parameters to TP ranks for async TP scenario.

        This map is used to determine which TP rank is responsible for each parameter's
        optimizer state, allowing us to only allocate dummy state for parameters owned
        by the current rank.
        """
        self.param_to_tp_rank_map = {}

        # Get executor from optimizer
        executor = self.optimizer.tp_param_group_executor

        # Build mapping for each param group that uses TP
        for group in self.optimizer.param_groups:
            use_matrix_based_optim = is_matrix_based_optim_group(group)
            if not use_matrix_based_optim:
                raise ValueError('for async tp should be matrix_based_optim')
            if not group['is_tensor_parallel']:
                continue
            # Get params and shapes for this group
            params = group['params']
            if 'origin_shape' in group:
                shapes = group['origin_shape']
            else:
                shapes = [p.shape for p in params]

            # Get param_to_tp_rank_map from executor
            group_map = executor.build_param_to_tp_rank_map(params, shapes)
            self.param_to_tp_rank_map.update(group_map)

    def _is_param_owned_by_current_tp_rank(self, model_param):
        """
        Check if the parameter is owned by the current TP rank in async TP scenario.

        Args:
            model_param: The model parameter.

        Returns:
            True if the current TP rank owns this parameter, False otherwise.
        """
        if not self.config.use_tp_async_opt:
            return True

        if not self.param_to_tp_rank_map:
            return True

        from megatron.core.parallel_state import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()

        # Get the global rank of the current TP rank
        from megatron.core.parallel_state import get_tensor_model_parallel_group
        from torch.distributed import get_global_rank
        tp_group = get_tensor_model_parallel_group()
        current_global_rank = get_global_rank(tp_group, tp_rank)

        # Check if this parameter is owned by current rank
        owned_rank = self.param_to_tp_rank_map.get(model_param)
        if owned_rank is None:
            # Parameter not in map, assume owned by all ranks (shouldn't happen for TP params)
            return True
        return owned_rank == current_global_rank

    def _should_allocate_state_for_param(self, model_param, state_order=None):
        """
        Determine if we should allocate dummy state for the given parameter.
        Returns:
            True if we should allocate dummy state, False otherwise.
        """
        return self._is_param_owned_by_current_tp_rank(model_param)

    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """

        inner_state_dict = self.optimizer.state_dict()
        state_dict = {}

        # Extract 'step', for non-Apex/TE support.
        if not HAVE_APEX_OR_TE and self.use_optimizer == 'adam':
            steps = list(set([s["step"].item() for s in inner_state_dict["state"].values()]))
            assert len(steps) == 1
            step = steps[0]
        elif isinstance(self.optimizer, HybridDeviceOptimizer):
            step = None
            for optimizer in self.optimizer.sub_optimizers:
                if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                    if len(optimizer.state) == 0:
                        continue
                    steps = list(set([s["step"].item() for s in optimizer.state.values()]))
                    assert len(steps) == 1, f"steps: {optimizer.state}"
                    step = steps[0]
                    break
        elif USING_TE_OPTIMIZER or USING_APEX_OPTIMIZER:
            # Extract 'step', for TE FusedAdam support.
            steps = list(
                set(
                    [
                        g["step"]
                        for g in inner_state_dict["param_groups"]
                        if len(g["params"]) > 0 and "step" in g
                    ]
                )
            )
            assert len(steps) <= 1, f"steps: {steps}"
            step = steps[0] if len(steps) == 1 else None
        
        # Optimizer state (do not store parameter state here).
        state_dict['optimizer'] = {k: v for k, v in inner_state_dict.items() if k != "state"}
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]
            if not HAVE_APEX_OR_TE and self.use_optimizer == 'adam':
                # Native PyTorch param group requires step (i.e., iteration).
                param_group["step"] = step
            elif (
                USING_TE_OPTIMIZER
                or USING_APEX_OPTIMIZER
                or isinstance(self.optimizer, HybridDeviceOptimizer)
            ) and step is not None:
                # TE FusedAdam will not accumulate step for empty param groups, so we need to
                # align the step across param groups.
                param_group["step"] = int(step)

        # Grad scaler state.
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """
        if self.ddp_config.use_megatron_fsdp:
            if "param_to_group_meta" in state_dict:
                state_dict["param_groups"] = self._param2group_meta_to_param_groups(
                    state_dict["param_to_group_meta"], self.optimizer.param_groups
                )
                del state_dict["param_to_group_meta"]
            self.optimizer.load_state_dict(state_dict)
            return

        if len(self.optimizer.state) == 0:
            if isinstance(self.optimizer, HybridDeviceOptimizer):
                self.optimizer.dummy_step()

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer ordering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        def make_needed_groups(param_group):
            needed_groups = []
            for key in param_group_identifier_keys:
                # NeMo changes these variable names from `lr_mult` and `wd_mult`
                # to `pre_lr_mult` and `pre_wd_mult`, so we need to check both.
                if key in param_group:
                    pass
                elif f"pre_{key}" in param_group:
                    key = f"pre_{key}"
                else:
                    raise ValueError(
                        f"Key {key} (or pre_{key}) not found in param_group {param_group}."
                    )
                needed_groups.append(param_group[key])
            needed_groups = tuple(needed_groups)
            return needed_groups

        param_groups_map = {}
        for param_group in state_dict["optimizer"]["param_groups"]:
            needed_groups = make_needed_groups(param_group)
            param_groups_map[needed_groups] = param_group
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = []
        for inner_param_group in inner_state_dict["param_groups"]:
            needed_groups = make_needed_groups(inner_param_group)
            state_dict_param_groups.append(
                {**param_groups_map[needed_groups], "params": inner_param_group['params']}
            )

        # Allocate or retrieve optimizer state (i.e., tensors).
        if len(self.optimizer.state) == 0:
            # Allocate empty optimizer state if not previously initialized.
            # - If len(self.optimizer.state) == 0, this means that the optimizer
            #   state has not been previously initialized. Once it has been
            #   initialized, we skip this code block to avoid reallocating
            #   empty tensors (i.e., torch.empty), which in turn reduces memory
            #   fragmentation.
            # - Real data is overwritten during load_parameter_state().
            state_dict_state = []
            for gbuf_range_maps in self.gbuf_ranges:
                for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                    for gbuf_range_map in gbuf_range_map_for_all_buckets:
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():

                            # Get parameter ordering information (see method docstring
                            # for details).
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            state_order = inner_state_dict["param_groups"][group_index]["params"][
                                group_order
                            ]
                            is_tensor_parallel = self.optimizer.param_groups[group_index]['is_tensor_parallel']
                            # In async TP scenario, only allocate state for parameters owned by
                            # current TP rank to reduce memory usage.
                            main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                            if not self._should_allocate_state_for_param(main_param):
                                state_dict_state.append((state_order, {}))
                                continue

                            # Allocate dummy tensors.
                            numel = len(param_range_map["gbuf_world"])
                            init_shard = lambda dtype=torch.float32: torch.empty(
                                (numel,), dtype=dtype, device=torch.cuda.current_device()
                            )
                            if is_tensor_parallel:
                                numel *= get_tensor_model_parallel_world_size()
                            init_shard = lambda: torch.empty(
                                (numel,), dtype=torch.float32, device=torch.cuda.current_device()
                            )
                            if self.use_optimizer == 'adam':
                                tensors = {
                                    "exp_avg": init_shard(self.config.exp_avg_dtype),
                                    "exp_avg_sq": init_shard(self.config.exp_avg_sq_dtype),
                                }
                            elif self.use_optimizer == 'muon':
                                # Check if this parameter has split attributes
                                split_methods = self.optim_splitter.get_split_param_methods(model_param) if self.optim_splitter is not None else []
                                if split_methods:
                                    # Handle split parameters
                                    assert len(split_methods) == 1, f"Only one split method supported, got {split_methods}"
                                    split_method = split_methods[0]

                                    # Get the gradient shapes for each split fragment
                                    split_grad_shapes = self.optim_splitter.compute_split_grad_shapes(
                                        model_param, split_method
                                    )
                                    muon_state_dict = {}
                                    # Create state for each split fragment
                                    for frag_idx, frag_shape in enumerate(split_grad_shapes):
                                        prefix = f"{split_method}.{frag_idx}."
                                        # momentum_buffer shape should match the fragment gradient shape (flattened)
                                        frag_numel = torch.Size(frag_shape).numel()
                                        frag_init = lambda: torch.empty(
                                            (frag_numel,), dtype=torch.float32, device=torch.cuda.current_device()
                                        )
                                        muon_state_dict[f"{prefix}momentum_buffer"] = frag_init()
                                    tensors = muon_state_dict
                                else:
                                    # Handle non-split parameters (original logic)
                                    tensors = {"momentum_buffer": init_shard(), }
                            elif self.use_optimizer == 'soap':
                                # Check if this parameter has split attributes
                                split_methods = self.optim_splitter.get_split_param_methods(model_param) if self.optim_splitter is not None else []

                                if split_methods:
                                    # Handle split parameters
                                    assert len(split_methods) == 1, f"Only one split method supported, got {split_methods}"
                                    split_method = split_methods[0]

                                    # Get the gradient shapes for each split fragment
                                    split_grad_shapes = self.optim_splitter.compute_split_grad_shapes(
                                        model_param, split_method
                                    )
                                    soap_state_dict={}
                                    # Create state for each split fragment
                                    for frag_idx, frag_shape in enumerate(split_grad_shapes):
                                        prefix = f"{split_method}.{frag_idx}."
                                        # exp_avg and exp_avg_sq shape should match the fragment gradient shape (flattened)
                                        frag_numel = torch.Size(frag_shape).numel()
                                        frag_init = lambda: torch.empty(
                                            (frag_numel,), dtype=torch.float32, device=torch.cuda.current_device()
                                        )
                                        soap_state_dict[f"{prefix}exp_avg"] = frag_init()
                                        soap_state_dict[f"{prefix}exp_avg_sq"] = frag_init()

                                        for idx, sh in enumerate(frag_shape):
                                            if sh > self.config.soap_max_precond_dim:
                                                pass
                                            else:
                                                soap_state_dict[f"{prefix}GG_{idx}"] = torch.empty(
                                                    (sh*sh,), dtype=torch.float32, device=torch.cuda.current_device()
                                                )
                                                soap_state_dict[f"{prefix}Q_{idx}"] = torch.empty(
                                                    (sh*sh,), dtype=torch.float32, device=torch.cuda.current_device()
                                                )
                                        soap_state_dict[f"{prefix}step"] = torch.empty(
                                            (1,), dtype=torch.int64, device=torch.cuda.current_device()
                                        )
                                    state_dict_state.append((state_order, soap_state_dict))
                                else:
                                    # Handle non-split parameters (original logic)
                                    soap_state_dict = {"exp_avg": init_shard(), "exp_avg_sq": init_shard(),}
                                    param_shape = list(model_param.shape)
                                    if is_tensor_parallel:
                                        scale = get_tensor_model_parallel_world_size()
                                        dim = getattr(model_param, 'partition_dim', None)
                                        param_shape[dim] *= scale
                                    for idx, sh in enumerate(param_shape):
                                        if sh > self.config.soap_max_precond_dim:
                                            pass
                                        else:
                                            soap_state_dict[f'GG_{idx}'] = torch.empty(
                                                (sh*sh,), dtype=torch.float32, device=torch.cuda.current_device()
                                            )
                                            soap_state_dict[f'Q_{idx}'] = torch.empty(
                                                (sh*sh,), dtype=torch.float32, device=torch.cuda.current_device()
                                            )
                                    soap_state_dict["step"] = torch.empty(
                                        (1,), dtype=torch.int64, device=torch.cuda.current_device()
                                    )
                                    tensors = soap_state_dict
                            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                                if self.config.store_param_remainders and self.config.bf16:
                                    tensors["master_param"] = init_shard(torch.int16)
                                else:
                                    tensors["master_param"] = init_shard(
                                        self.config.main_params_dtype
                                    )
                            state_dict_state.append((state_order, tensors))


            # Sort by state order (see method docstring for details).
            state_dict_state.sort(key=lambda s: s[0])
            state_dict_state = {s[0]: s[1] for s in state_dict_state}

        else:
            # Retrieve existing optimizer state.
            state_dict_state = inner_state_dict["state"]
        # Extract 'step', for non-Apex/TE support.
        if not HAVE_APEX_OR_TE and self.use_optimizer == 'adam':
            steps = list(set([g["step"] for g in state_dict["optimizer"]["param_groups"]]))
            assert len(steps) == 1
            step = torch.tensor(steps[0], dtype=torch.float)

            for s in state_dict_state.values():
                # Native PyTorch state dict requires step (i.e., iteration).
                s["step"] = step
        elif isinstance(self.optimizer, HybridDeviceOptimizer):
            # Handle Torch AdamW special case, which, unlike FusedAdam, Torch AdamW
            # has an extra optimizer state "step".
            steps = list(
                set([g["step"] for g in state_dict["optimizer"]["param_groups"] if "step" in g])
            )
            if len(steps) != 0:
                assert len(steps) == 1, f"steps: {steps}"
                step = torch.tensor(steps[0], dtype=torch.float32, device="cpu")
                for v in self.optimizer.state.values():
                    v["step"] = step.detach().clone()

        # Empty cuda cache to avoid CUDA OOM.
        torch.cuda.empty_cache()

        # Optimizer.
        self.optimizer.load_state_dict(
            {"state": state_dict_state, "param_groups": state_dict_param_groups}
        )
        # recast dtype to int for soap
        if self.use_optimizer == 'soap':
            for _tensor, _state_dict in self.optimizer.state.items():
                for state_key, shared_state in _state_dict.items():
                    if state_key in ["step"]:
                        self.optimizer.state[_tensor][state_key] = shared_state.to(torch.int64)

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                log_single_rank(
                    logger,
                    logging.INFO,
                    '***WARNING*** found an old checkpoint, will not load grad scaler ...',
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                log_single_rank(
                    logger,
                    logging.INFO,
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...',
                )

        if 'param_state' in state_dict:
            assert 'param_state_sharding_type' in state_dict, state_dict.keys()
            param_state = state_dict['param_state']
            sharding_type = state_dict['param_state_sharding_type']
            log_single_rank(
                logger,
                logging.INFO,
                f'Loading distributed optimizer sharded state of type {sharding_type}',
            )
            if sharding_type == 'dp_zero_gather_scatter':
                assert self.use_optimizer == 'adam', 'only adam optimizer support legacy zero'
                self.load_parameter_state_from_dp_zero(param_state)
            elif sharding_type == 'fully_reshardable':
                raise NotImplementedError
                self.load_parameter_state_from_fully_reshardable(param_state)
            elif sharding_type == 'dp_reshardable':
                raise NotImplementedError
                self.load_parameter_state_from_dp_reshardable(param_state)
            elif sharding_type == 'fully_sharded_model_space':
                self.load_parameter_state_from_fs_model_space(param_state)
            else:
                raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict = {},
        is_loading: bool = False,
        sharding_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Chooses between 3 param state sharding implementations as requested by `sharding_type`.

        Regular state dict parameters are saved on DP rank 0 and loaded on all ranks.
        """
        sharding_type = 'fully_sharded_model_space'

        if sharding_type is not None:
            log_single_rank(
                logger,
                logging.WARNING,
                'DistributedOptimizer.sharded_state_dict parameter `sharding_type`'
                ' is deprecated and will be removed.'
                ' Use `metadata["distrib_optim_sharding_type"] instead`.',
            )
        else:
            sharding_type = (metadata or {}).get(
                'distrib_optim_sharding_type', 'fully_sharded_model_space'
            )

        # Handle FSDP DistributedOptimizer States
        if self.ddp_config.use_megatron_fsdp and sharding_type != "fsdp_dtensor":
            raise NotImplementedError(
                f"sharding_type {sharding_type} is not supported with Megatron FSDP."
            )
        if sharding_type == "fsdp_dtensor":
            state_dict = self.sharded_param_state_fsdp_dtensor(is_loading)
            return state_dict

        if not is_loading and sharding_type == 'fully_sharded_bucket_space':
            logger.warning(
                '`fully_sharded_bucket_space` sharding for DistributedOptimizer'
                ' checkpoint is deprecated and will be removed in the future.'
                ' Please switch to `full_sharded_model_space`.'
            )

        state_dict = self.state_dict()
        if sharding_type != 'fully_sharded_model_space':
            # State dict differs between different model parallel groups
            state_dict = {
                k: ShardedObject(
                    f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.{k}',
                    v,
                    (1,),
                    (0,),
                    replica_id=(
                        self.distributed_optimizer_instance_id,
                        0,
                        self.data_parallel_group.rank(),
                    ),
                )
                for k, v in state_dict.items()
            }

        if is_loading:
            # Call the distributed optimizer's specialized load_state_dict(),
            # which conditionally skips re-allocating the optimizer's state if
            # already initialized, which in turn reduces memory fragmentation.
            self.load_state_dict(self.state_dict())
        if self.config.use_tp_async_opt or is_matrix_based_optim(self.use_optimizer):
            assert sharding_type == 'fully_sharded_model_space', f"for canzon, and matrix based optimizer only support fully_sharded_model_space sharding type for now. Got 'sharding_type'={sharding_type}"
        if sharding_type == 'dp_reshardable':
            param_state = self.sharded_param_state_dp_reshardable(
                model_sharded_state_dict, is_loading, metadata
            )

        elif sharding_type == 'dp_zero_gather_scatter':
            # NOTE: this format will be deprecated
            param_state = self.sharded_param_state_dp_zero(
                model_sharded_state_dict, is_loading, metadata
            )
            gc.collect()  # Prevent memory leaks with GC disabled
        elif sharding_type == 'fully_reshardable':
            raise NotImplementedError
            param_state = self.sharded_param_state_fully_reshardable(
                model_sharded_state_dict, is_loading, metadata
            )
            gc.collect()  # Prevent memory leaks with GC disabled
        elif sharding_type == 'fully_sharded_model_space':
            # NOTE: this format will be deprecated
            param_state = self.sharded_param_state_fs_model_space(
                model_sharded_state_dict, is_loading, metadata
            )
        else:
            raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

        state_dict['param_state'] = param_state
        state_dict['param_state_sharding_type'] = sharding_type
        return state_dict

    def sharded_param_state_fs_model_space(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        param_to_sharded_metadata = {}
        model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(
            model_sharded_state_dict
        )
        for sh_base in nested_values(model_sharded_state_dict):
            param_to_sharded_metadata[sh_base.data] = sh_base

        prefix = 'optimizer.state'
        state = {}

        def _get_param_state_sharded_tensors(model_param, item_slice):
            group_index, group_order = self.model_param_group_index_map[model_param]
            
            # Main param & optimizer states.
            tensors = self._get_main_param_and_optimizer_states(model_param)
            tensors["fp32_param"] = tensors.pop("param")

            try:
                sharded_metadata = param_to_sharded_metadata[model_param]
            except KeyError as e:
                raise ValueError(
                    f"Model param {model_param} not in model_sharded_state_dict"
                    f" Hint: {KEEP_VARS_HINT}"
                ) from e

            # Set DP corresponding replica_id coordinate to 0.
            assert (
                len(sharded_metadata.replica_id) == 3
            ), f'Expected replica_id format (PP, TP, DP), got: {sharded_metadata}'
            replica_id = (*sharded_metadata.replica_id[:2], self.distributed_optimizer_instance_id)

            for state_key, state_ten in tensors.items():
                if state_key == 'step':
                    continue

                regular_state = state_key == 'fp32_param'

                if regular_state:
                    replace_kwargs = dict(
                        key=f'{prefix}.{state_key}.{sharded_metadata.key}',
                        data=state_ten,
                        dtype=state_ten.dtype,
                        flattened_range=item_slice,
                        replica_id=replica_id,
                    )
                    if isinstance(sharded_metadata, ShardedTensorFactory):
                        replace_kwargs.pop('dtype')
                    tensors[state_key] = replace(sharded_metadata, **replace_kwargs)
                    tensors[state_key].validate_metadata_integrity()
                else:
                    state_sharded_metadata = sharded_metadata
                    if isinstance(sharded_metadata, ShardedTensorFactory):
                        state_sharded_metadata = sharded_metadata.build()[0]

                    if len(state_sharded_metadata.global_offset) == 3:
                        global_offset = state_sharded_metadata.global_offset[:-1]
                    else:
                        global_offset = (0,)

                    if len(state_sharded_metadata.axis_fragmentations) == 3:
                        axis_fragmentations = (state_sharded_metadata.axis_fragmentations[0], 1)
                    else:
                        axis_fragmentations = (1,)

                    if any(sub in state_key for sub in ['Q', 'GG', 'step']):
                        dim = math.isqrt(state_ten.numel())
                        assert dim * dim == state_ten.numel(), (
                            f'for {state_key=} {state_ten.numel()=} {dim=}'
                        )
                    
                    if not self.optimizer.param_groups[group_index]['is_tensor_parallel']:
                        matrix_replica_id = (0, sharded_metadata.replica_id[1], 0)
                    else:
                        matrix_replica_id = (0, 0, 0)

                    global_shape_prepend = state_sharded_metadata.global_shape[:-2]
                    local_shape = tuple(state_ten.shape)

                    replace_kwargs = dict(
                        key=f'{prefix}.{state_key}.{state_sharded_metadata.key}',
                        data=state_ten,
                        dtype=state_ten.dtype,
                        flattened_range=slice(0, state_ten.numel()),
                        replica_id=matrix_replica_id,
                        global_shape=global_shape_prepend + local_shape,
                        local_shape=local_shape,
                        axis_fragmentations=axis_fragmentations,
                        global_offset=global_offset,
                    )
                    tensors[state_key] = replace(state_sharded_metadata, **replace_kwargs)
                    tensors[state_key].validate_metadata_integrity()

            return tensors

        param_idx = 0
        for gbuf_range_maps in self.gbuf_ranges:
            for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        param_range = param_range_map['param']
                        tensors = _get_param_state_sharded_tensors(
                            model_param, slice(param_range.start, param_range.end)
                        )
                        state[param_idx] = tensors
                        param_idx += 1
        return state
