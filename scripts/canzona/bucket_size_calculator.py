"""
Calculator for determining optimal MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE.

Purpose:
    Find bucket_size values that make Megatron's regular expert buffer produce
    useful even-sharded buckets:
    1. Each bucket contains a whole number of expert params (no param split)
    2. Buckets whose expert-param count is evenly divisible by EDP can be split
       across EDP ranks by whole expert params and use the even communication
       fast path

    Ideally all buckets are even. When the expert buffer produces two or more
    buckets, however, a mixed layout is still useful: even buckets can use the
    high-performance path while uneven buckets keep using the compatible path.
    Therefore this calculator treats any bucket_size that creates at least one
    even bucket as useful, and it reports both even and uneven bucket counts.

How buckets form:
    Params are accumulated in reverse backprop order until the accumulated numel
    is greater than or equal to bucket_size, at which point a bucket is formed
    containing all accumulated complete params. Megatron also aligns each param
    start to a 64-element boundary, so the threshold for k params is:

        bucket_size = (k - 1) * align64(expert_param_numel) + expert_param_numel

    When expert_param_numel is already 64-aligned, this simplifies to:

        bucket_size = k * expert_param_numel

    For matrix-based optimizer expert buckets, the useful fast path is when the
    bucket can be divided across EDP ranks by whole expert params. Since every
    expert param has the same size, valid full-bucket sizes are:

        k = EDP, 2*EDP, 3*EDP, ...

    The final residual bucket is checked independently. It is even only when
    its expert-param count is also divisible by EDP.

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
        - Bucket boundaries should keep expert params whole
        - For a bucket to be "even": params_in_bucket % EDP == 0

    How to get expert_param_numel from logs:
        Look at the DDP bucket log output for the expert buffer.
        Each bucket lists params with their shapes. The expert param shape
        (e.g., [2048, 1536]) gives you the numel directly.
"""

from typing import Optional

PARAM_START_ALIGNMENT = 64


def align_up(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def compute_edp(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: Optional[int] = None,
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
    max_bucket_size: Optional[int] = None,
) -> None:
    """
    Print a report of useful bucket_size options.

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
    if world_size % (tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size) != 0:
        raise ValueError(
            "world_size must be divisible by TP * PP * CP: "
            f"{world_size} vs "
            f"{tensor_parallel_size} * {pipeline_model_parallel_size} * {context_parallel_size}"
        )
    if total_num_layers % pipeline_model_parallel_size != 0:
        raise ValueError(
            f"total_num_layers ({total_num_layers}) must be divisible by PP "
            f"({pipeline_model_parallel_size})"
        )
    if num_experts % expert_model_parallel_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by EP "
            f"({expert_model_parallel_size})"
        )
    if expert_param_numel <= 0:
        raise ValueError(f"expert_param_numel must be positive, got {expert_param_numel}")

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

    aligned_expert_stride = align_up(expert_param_numel, PARAM_START_ALIGNMENT)

    def bucket_threshold_for_k(k: int) -> int:
        return (k - 1) * aligned_expert_stride + expert_param_numel

    k_min = edp_size

    if max_bucket_size is None:
        max_bucket_size = bucket_threshold_for_k(total_num_experts)
    if max_bucket_size < 0:
        raise ValueError(f"max_bucket_size must be non-negative, got {max_bucket_size}")

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
    print(f"  Aligned expert stride:      {aligned_expert_stride:,}")
    print(f"  Total expert numel (stage): {total_expert_numel:,}")
    print(f"  max_bucket_size:            {max_bucket_size:,}")

    print(f"  k_min = EDP:                {k_min}")
    print(f"  (k_min = minimum params per bucket so each EDP rank gets whole params)")

    print(f"\n{'='*80}")
    # k < k_min: too few params to give every EDP rank whole expert params.
    # k > max_k: threshold too large (beyond configured max_bucket_size).
    #
    # Megatron forms a bucket when accumulated_numel >= bucket_size after adding
    # the current param. Therefore setting bucket_size to the k-param threshold
    # makes the actual bucket contain k params.
    max_k = 0
    for k in range(1, total_num_experts + 1):
        if bucket_threshold_for_k(k) <= max_bucket_size:
            max_k = k
        else:
            break
    MAX_EXCEEDED = 5

    print(
        f"{'num_buckets':<14} "
        f"{'params/bucket':<15} "
        f"{'bucket_size':<20} "
        f"{'even/uneven':<14} "
        f"{'Status':<12} "
        f"{'Reason':<50}"
    )
    print(f"{'='*80}")

    # Show the last few below-min entries, all entries within max_bucket_size,
    # plus the first few above max.
    display_start = max(1, k_min - MAX_EXCEEDED)
    max_display_k = max_k + MAX_EXCEEDED
    useful_entries = []

    for k in range(display_start, min(total_num_experts, max_display_k) + 1):
        bucket_size = bucket_threshold_for_k(k)
        num_full_buckets = total_num_experts // k
        remainder_k = total_num_experts % k
        bucket_param_counts = [k] * num_full_buckets
        if remainder_k:
            bucket_param_counts.append(remainder_k)

        bucket_numels = [bucket_threshold_for_k(count) for count in bucket_param_counts]
        even_flags = [count % edp_size == 0 for count in bucket_param_counts]
        num_buckets = len(bucket_param_counts)
        total_bucket_numel = sum(bucket_numels)
        even_buckets = sum(even_flags)
        uneven_buckets = num_buckets - even_buckets
        even_bucket_numel = sum(
            numel for numel, is_even_bucket in zip(bucket_numels, even_flags) if is_even_bucket
        )
        has_even_bucket = even_buckets > 0
        all_buckets_even = has_even_bucket and uneven_buckets == 0
        below_min = k < k_min
        above_max = k > max_k
        if below_min:
            reason = f"params/bucket ({k}) < EDP ({edp_size}), not enough whole params"
            status = "TOO SMALL"
        elif above_max:
            reason = f"bucket_size ({bucket_size:,}) > max_bucket_size ({max_bucket_size:,})"
            status = "OUT OF RANGE"
        elif all_buckets_even:
            reason = "all buckets can use even fast path"
            status = "ALL EVEN"
        elif has_even_bucket:
            reason = (
                f"{even_buckets} bucket(s) can use even fast path; "
                f"{uneven_buckets} bucket(s) remain uneven"
            )
            status = "PARTIAL"
        else:
            reason = f"no bucket expert-param count is divisible by EDP ({edp_size})"
            status = "NO"
        print(
            f"{num_buckets:<14} "
            f"{k:<15} "
            f"{bucket_size:<20,} "
            f"{even_buckets}/{uneven_buckets:<12} "
            f"{status:<12} "
            f"{reason}"
        )
        if not below_min and not above_max and has_even_bucket:
            useful_entries.append(
                (
                    all_buckets_even,
                    even_bucket_numel,
                    -num_buckets,
                    k,
                    bucket_size,
                    even_buckets,
                    num_buckets,
                    uneven_buckets,
                    total_bucket_numel,
                )
            )

    if total_num_experts > max_display_k:
        print("...")

    print(f"\n{'='*80}")

    if display_start > 1:
        print(f"... (and {display_start - 1} more entries below k_min={k_min})")

    if not useful_entries:
        print("WARNING: No useful bucket_size found within [min, max] range!")
        print(f"  k_min = EDP = {k_min}, max_k = {max_k}")
        if k_min > max_k:
            print(f"  You need k_min <= k <= max_k, but {k_min} > {max_k}.")
            print(f"  Either increase max_bucket_size or decrease EDP size.")
        else:
            print("  No displayed k creates a bucket whose expert-param count is divisible by EDP.")
    else:
        # Prefer all-even layouts. Otherwise maximize the amount of bucket data
        # covered by even buckets, then prefer fewer buckets and larger buckets.
        (
            all_even_rec,
            even_bucket_numel_rec,
            _neg_num_buckets_rec,
            k_rec,
            bucket_size_rec,
            even_buckets_rec,
            actual_buckets_rec,
            uneven_buckets_rec,
            total_bucket_numel_rec,
        ) = max(useful_entries, key=lambda entry: entry[:8])
        print(f"Recommended (max_bucket_size={max_bucket_size:,}):")
        print(f"  export MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE={bucket_size_rec}")
        print(
            f"  (full buckets have {k_rec} params, produces {actual_buckets_rec} bucket(s): "
            f"{even_buckets_rec} even, {uneven_buckets_rec} uneven)"
        )
        if not all_even_rec:
            print(
                f"  Partial fast path covers {even_bucket_numel_rec:,}/"
                f"{total_bucket_numel_rec:,} real bucket elements."
            )

    print(f"  Useful search range: k >= {k_min}, k <= {max_k}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example: MoE 7B-A1B
    print_bucket_size_report(
        world_size=64,    # number of GPUs
        tensor_parallel_size=4, # TP size
        pipeline_model_parallel_size=1, # PP size
        context_parallel_size=1,    # CP size
        expert_model_parallel_size=8,   # EP size
        total_num_layers=28,
        num_experts=64,
        expert_param_numel=4718592, # param numel per expert: hidden_size * moe_ffn * 3
        max_bucket_size=800000000,  # allow up to ~800M element per bucket
    )
