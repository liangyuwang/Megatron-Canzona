"""
Calculator for determining optimal MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE.

Purpose:
    Find bucket_size values that ensure:
    1. Each bucket contains a whole number of expert params (no param split)
    2. Each bucket's numel is evenly divisible by EDP (even sharding, fast path)

How buckets form:
    Params are accumulated in reverse backprop order until the accumulated numel
    exceeds bucket_size, at which point a bucket is formed containing all accumulated
    (complete) params. So each bucket contains k_params complete params where
    k_params ≈ bucket_size / expert_param_numel.

    A bucket is "even" when bucket_numel % EDP == 0, i.e. each DP rank gets the same
    number of elements. Since each expert param has the same size, this means:
    k_params * expert_param_numel % EDP == 0

    k_min = EDP is the minimum number of params per bucket to guarantee even sharding.
    After ReduceScatter, each rank receives (k * expert_param_numel / EDP) elements.
    For each rank to hold whole params (not a split param), we need:
        (k * expert_param_numel / EDP) % expert_param_numel == 0
        => k % EDP == 0
    So valid bucket sizes are k = EDP, 2*EDP, 3*EDP, ...
    and k_min = EDP.

Parallelism decomposition:
    world_size = TP * PP * CP * DP

    Users set: TP, PP, CP, DP, EP (expert model parallel)
    EDP is derived:

        expert_pipeline_size = ETP * EP * PP  (ETP defaults to TP)
        EDP = world_size / expert_pipeline_size = CP * DP / EP

        EP = 1: EDP = CP * DP  (expert 和 dense 用同一个 DP group)
        EP > 1: EDP = CP * DP / EP  (expert 看到的 DP group 更小)

    Why this matters:
        - Dense buffers use DP as their all-reduce/reduce-scatter group
        - Expert buffers use EDP as their group
        - Matrix-based optimizer requires each param to be whole (not split across ranks)
        - For a bucket to be "even": bucket_numel % EDP == 0

    How to get expert_param_numel from logs:
        Look at the DDP bucket log output for the expert buffer.
        Each bucket lists params with their shapes. The expert param shape
        (e.g., [2048, 1536]) gives you the numel directly.
"""

import math


def compute_edp(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: int | None = None,
) -> int:
    """
    Compute EDP (expert data parallel) from user-set parameters.

    EDP = world_size / (ETP * EP * PP)
    When expert_tensor_parallel_size is not set, it defaults to tensor_parallel_size.
    """
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_parallel_size
    expert_pipeline_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    assert world_size % expert_pipeline_size == 0, (
        f"world_size ({world_size}) not divisible by expert pipeline size ({expert_pipeline_size})"
    )
    return world_size // expert_pipeline_size


def print_bucket_size_report(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    total_num_layers: int,
    num_experts: int,
    expert_param_numel: int,
    max_bucket_size: int | None = None,
) -> None:
    """
    Print a report of valid bucket_size options.

    Args:
        world_size: total GPU count
        tensor_parallel_size: TP
        pipeline_model_parallel_size: PP
        context_parallel_size: CP
        expert_model_parallel_size: EP
        total_num_layers: total number of transformer layers
        num_experts: number of experts per layer (per PP stage)
        expert_param_numel: numel of a single expert param matrix
        max_bucket_size: maximum bucket_size to consider (defaults to total_numel)
    """
    dp_size = world_size // (
        tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    edp_size = compute_edp(
        world_size,
        tensor_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        expert_model_parallel_size,
    )

    num_layers_per_stage = total_num_layers // pipeline_model_parallel_size
    experts_per_ep_rank = num_experts // expert_model_parallel_size
    total_num_experts = experts_per_ep_rank * num_layers_per_stage
    total_expert_numel = experts_per_ep_rank * num_layers_per_stage * expert_param_numel

    k_min = edp_size

    if max_bucket_size is None:
        max_bucket_size = total_expert_numel

    print("=" * 80)
    print("MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE Calculator")
    print("=" * 80)

    print(f"\nInput parameters:")
    print(f"  world_size:                 {world_size}")
    print(f"  TP:                         {tensor_parallel_size}")
    print(f"  PP:                         {pipeline_model_parallel_size}")
    print(f"  CP:                         {context_parallel_size}")
    print(f"  DP (=W/TP/PP/CP):           {dp_size}")
    print(f"  EP:                         {expert_model_parallel_size}")
    print(f"  EDP (=W/ETP/EP/PP):         {edp_size}")
    print(f"  Total layers:               {total_num_layers}")
    print(f"  Layers per PP stage:        {num_layers_per_stage}")
    print(f"  Experts per layer:          {num_experts}")
    print(f"  Experts per EP rank:        {experts_per_ep_rank}")
    print(f"  Total experts (per rank):   {total_num_experts}")
    print(f"  Expert param numel:         {expert_param_numel:,}")
    print(f"  Total expert numel (stage): {total_expert_numel:,}")
    print(f"  max_bucket_size:            {max_bucket_size:,}")

    print(f"  k_min = EDP = {k_min}")
    print(f"  (k_min = minimum params per bucket so every EDP rank gets a whole param)")

    print(f"\n{'='*80}")
    # k < k_min: not enough params to give each EDP rank a whole param.
    # k > max_k: bucket too large (beyond configured max_bucket_size).
    max_k = max_bucket_size // expert_param_numel
    MAX_EXCEEDED = 5

    print(f"{'num_buckets':<14} {'params/bucket':<15} {'bucket_size':<20} {'Even?':<8} {'Reason':<50}")
    print(f"{'='*80}")

    # Show: last 5 below_min entries, all valid entries, first 5 above_max entries.
    below_min_start = max(1, k_min - MAX_EXCEEDED)
    max_display_k = max_k + MAX_EXCEEDED
    valid_entries = []

    for k in range(below_min_start, min(total_num_experts, max_display_k) + 1):
        bucket_size = k * expert_param_numel
        num_buckets = math.ceil(total_expert_numel / bucket_size)
        is_even = (k % edp_size == 0)
        below_min = k < k_min
        above_max = k > max_k
        if below_min:
            reason = f"params/bucket ({k}) < EDP ({edp_size}), not enough to shard"
        elif above_max:
            reason = f"bucket_size ({bucket_size:,}) > max_bucket_size ({max_bucket_size:,})"
        elif is_even:
            reason = "OK"
        else:
            reason = f"params/bucket ({k}) not a multiple of EDP ({edp_size}), uneven sharding"
        print(
            f"{num_buckets:<14} "
            f"{k:<15} "
            f"{bucket_size:<20,} "
            f"{'YES' if is_even else 'NO':<8} "
            f"{reason}"
        )
        if not below_min and not above_max and is_even:
            valid_entries.append((num_buckets, k, bucket_size))

    if total_num_experts > max_display_k:
        print("...")

    if below_min_start > 1:
        print(f"... (and {below_min_start - 1} more entries below k_min={k_min})")

    print(f"\n{'='*80}")

    if not valid_entries:
        print("WARNING: No valid bucket_size found within [min, max] range!")
        print(f"  k_min (EDP) = {k_min}, max_k = {max_k}")
        print(f"  You need k_min <= k <= max_k, but {k_min} > {max_k}.")
        print(f"  Either increase max_bucket_size or decrease EDP size.")
    else:
        # Recommend the largest valid k (fewest buckets).
        _, k_rec, bucket_size_rec = max(valid_entries, key=lambda x: x[1])
        actual_buckets_rec = math.ceil(total_expert_numel / bucket_size_rec)
        print(f"Recommended (max_bucket_size={max_bucket_size:,}):")
        print(f"  export MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE={bucket_size_rec}")
        print(f"  (each bucket has {k_rec} params, produces {actual_buckets_rec} bucket(s))")

    print(f"  Valid range: k >= {k_min}, k <= {max_k}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example: Turbo Size
    print_bucket_size_report(
        world_size=3072,    # number of GPUs
        tensor_parallel_size=1, # TP size
        pipeline_model_parallel_size=2, # PP size
        context_parallel_size=1,    # CP size
        expert_model_parallel_size=8,   # EP size
        total_num_layers=28,
        num_experts=256,
        expert_param_numel=3145728, # param numel per expert: ffn * moe_ffn * 3
        max_bucket_size=256 * 3145728,  # allow up to ~800M element per bucket
    )
