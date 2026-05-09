import os
import logging
from typing import List, Tuple, Dict, Callable

from megatron.training import get_args
from megatron.core.utils import log_on_each_pipeline_stage
from .utils import (
    get_optim_memory_from_param, get_optim_flops_from_param,
)

logger = logging.getLogger(__name__)

def get_balance_ratio(arr: list[int]) -> float:
    if sum(arr) == 0:
        return 1.0
    return max(arr) / (sum(arr) / len(arr)) if arr else 1.0

def bucket_rank_slices(bucket_params_range_map: dict, bucket_slices_pos: list, dp_size: int) -> Tuple[int, List[int]]:
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if len(bucket_params_range_map) <= 0:
        raise ValueError(f"length of bucket_params_range_map {len(bucket_params_range_map)} must be positive")
    if len(bucket_slices_pos)-1 != dp_size:
        raise ValueError(f"length of bucket_slices_pos: {len(bucket_slices_pos)} must be equal to dp_size+1: {dp_size+1}")

    param_rank_map = {}
    rank_param_map = {r: [] for r in range(dp_size)}
    for i, (param, p_range) in enumerate(bucket_params_range_map.items()):
        for r in range(len(bucket_slices_pos)-1):
            if p_range[0] >= bucket_slices_pos[r] and p_range[0] < bucket_slices_pos[r+1]:
                param_rank_map[param] = r
                rank_param_map[r].append(param)
            if p_range[0] < bucket_slices_pos[r+1] and p_range[1] >= bucket_slices_pos[r+1]:
                raise ValueError(f"Slicing a param ({p_range}) at point {bucket_slices_pos[r+1]}")
    return param_rank_map, rank_param_map

def compute_objectives(loads: List[int]) -> Dict[str, float]:
    R = len(loads)
    total = sum(loads)
    mu = total / R if R else 0.0
    makespan = max(loads) if loads else 0
    max_abs_dev = max(abs(L - mu) for L in loads) if loads else 0.0
    var = sum((L - mu) ** 2 for L in loads) / R if R else 0.0
    return {
        "total": float(total),
        "mean": float(mu),
        "makespan": float(makespan),
        "max_abs_dev": float(max_abs_dev),
        "variance": float(var),
    }

def bucket_rank_loads(bucket_params_range_map: dict, bucket_slices_pos: list, dp_size: int) -> List[int]:
    _, per_rank = bucket_rank_slices(bucket_params_range_map, bucket_slices_pos, dp_size)
    return [sum(dp_balance_cost(p) for p in per_rank[r]) if r in per_rank else 0 for r in range(dp_size)]

def global_rank_loads(buckets_params_range_map: dict, buckets_slices_pos: list, dp_size: int) -> List[int]:
    loads = [0] * dp_size
    for bucket_params_range_map, bucket_slices_pos in zip(buckets_params_range_map, buckets_slices_pos):
        br = bucket_rank_loads(bucket_params_range_map, bucket_slices_pos, dp_size)
        for r in range(dp_size):
            loads[r] += br[r]
    return loads


# =================================
# ------- optim algorithm --------
# =================================

def greedy_lpt_with_ranges(
    buckets_params_range_map: List[Dict],
    dp_size: int,
    explicit_bucket_totals: List[int],
    bucket_order: str = "by_total_desc",
    alpha: float = 1.0  # alpha: 0.0 (comm balance) <-> 1.0 (DP balance)
) -> List[List[int]]:
    """
    Add trade-off between DP balance and comm balance
    """
    # Pre-calculate Metrics & Sort
    n_buckets = len(buckets_params_range_map)
    bucket_infos = [] 
    total_global_numel = 0
    for i, b_map in enumerate(buckets_params_range_map):
        boundaries = [0]
        for _, rng in b_map.items():
            boundaries.append(rng[1] + 1)
        b_total = explicit_bucket_totals[i]
        if b_total not in boundaries:
            boundaries.append(b_total)
        boundaries.sort()
        total_global_numel += b_total
        bucket_infos.append({"idx": i, "total": b_total, "boundaries": boundaries})

    global_target_load = total_global_numel / dp_size
    if bucket_order == "by_total_desc":
        bucket_infos.sort(key=lambda x: x["total"], reverse=True)
    
    # Allocation Loop
    current_rank_loads = [0] * dp_size
    results = [None] * n_buckets

    for b_info in bucket_infos:
        b_total = b_info["total"]
        boundaries = b_info["boundaries"]
        
        # Perfect Comm Balance
        even_share = b_total / dp_size
        base_allocs = [even_share] * dp_size
        
        # Perfect DP Balance Strategy
        deficits = [max(0, global_target_load - L) for L in current_rank_loads]
        total_deficit = sum(deficits)
        
        if total_deficit <= 1e-9:
            greedy_allocs = base_allocs
        else:
            greedy_allocs = [(d / total_deficit) * b_total for d in deficits]
            
        # Blending
        # alpha = 1.0 -> greedy_allocs
        # alpha = 0.0 -> base_allocs
        target_allocs = []
        for r in range(dp_size):
            mixed = (1 - alpha) * base_allocs[r] + alpha * greedy_allocs[r]
            target_allocs.append(mixed)
            
        # Discretize
        slice_pos = [0] * (dp_size + 1)
        slice_pos[0] = 0
        slice_pos[-1] = b_total
        current_cumulative_target = 0
        
        for r in range(dp_size - 1):
            current_cumulative_target += target_allocs[r]
            
            # Find nearest boundary (Scanning optimization omitted for brevity)
            best_boundary = boundaries[0]
            min_diff = abs(current_cumulative_target - best_boundary)
            for val in boundaries:
                if val < slice_pos[r]: continue
                diff = abs(current_cumulative_target - val)
                if diff < min_diff:
                    min_diff = diff
                    best_boundary = val
                else:
                    if val > current_cumulative_target: break
            slice_pos[r+1] = best_boundary
            current_rank_loads[r] += (slice_pos[r+1] - slice_pos[r])

        current_rank_loads[dp_size-1] += (slice_pos[dp_size] - slice_pos[dp_size-1])
        results[b_info["idx"]] = slice_pos

    return results

# =================================
# ------ balance_cost func -------
# =================================

def get_balance_cost_fn(balance_cost_name="numel") -> Callable:
    if balance_cost_name == "numel":
        return lambda param: param.numel()
    if balance_cost_name == "memory":
        return lambda param: get_optim_memory_from_param(param)
    elif balance_cost_name == "flops":
        return lambda param: get_optim_flops_from_param(param)
    else:
        raise ValueError(f"Unsupported balance_cost: {balance_cost_name}")

# =================================
# ------- visual utility --------- 
# =================================

def _human(n: int) -> str:
    # compact int formatting
    if n >= 10**9: return f"{n/1e9:.2f}B"
    if n >= 10**6: return f"{n/1e6:.2f}M"
    if n >= 10**3: return f"{n/1e3:.2f}K"
    return str(n)

def _bar(value: int, max_value: int, width: int = 40, ch: str = "█") -> str:
    if max_value <= 0:
        return ""
    filled = int(round(width * (value / max_value)))
    filled = max(0, min(width, filled))
    return ch * filled + " " * (width - filled)

def _alloc_proportional(length: int, weights: List[int]) -> List[int]:
    """Allocate `length` chars proportionally to `weights` (sum of alloc == length)."""
    if length <= 0 or not weights:
        return [0] * len(weights)
    s = sum(weights)
    if s <= 0:
        return [0] * len(weights)

    raw = [w * length / s for w in weights]
    alloc = [int(x) for x in raw]
    rem = length - sum(alloc)

    frac_order = sorted(range(len(weights)), key=lambda i: (raw[i] - alloc[i]), reverse=True)
    for i in frac_order[:rem]:
        alloc[i] += 1
    return alloc

def visual_bucket_rank_loads(bucket_params_range_map: dict, bucket_slices_pos: list, dp_size: int, cost_fn: Callable) -> List[int]:
    _, per_rank = bucket_rank_slices(bucket_params_range_map, bucket_slices_pos, dp_size)
    return [sum(cost_fn(p) for p in per_rank[r]) if r in per_rank else 0 for r in range(dp_size)]

def visual_global_rank_loads(buckets_params_range_map: dict, buckets_slices_pos: list, dp_size: int, cost_fn: Callable) -> List[int]:
    loads = [0] * dp_size
    for bucket_params_range_map, bucket_slices_pos in zip(buckets_params_range_map, buckets_slices_pos):
        br = visual_bucket_rank_loads(bucket_params_range_map, bucket_slices_pos, dp_size, cost_fn)
        for r in range(dp_size):
            loads[r] += br[r]
    return loads

def _log(msg: str) -> None:
    log_on_each_pipeline_stage(logger, logging.INFO, msg)

def log_bucket_slice_bars(
    buckets_params_range_map: List[Dict],
    buckets_slices_pos: List[List[int]],
    dp_size: int,
    title: str = "",
    width: int = 48,
) -> None:
    """
    Visualizes how each bucket is sliced and distributed across DP ranks.
    - Each bar represents a bucket.
    - The length of the bar is proportional to the bucket's total numel.
    - The numbers (0, 1, 2, ...) inside the bar represent DP ranks.
    - The length of each number segment shows the proportion of the bucket assigned to that rank.
    """
    if title:
        _log(f"\n== {title} ==")

    # Total numel of each bucket is the last slice position.
    totals = [bsp[-1] for bsp in buckets_slices_pos]
    max_total = max(totals) if totals else 0

    for b_id, bucket_map in enumerate(buckets_params_range_map):
        total = totals[b_id]
        n_params = len(bucket_map)
        
        if max_total <= 0 or total <= 0:
            scaled_len = 0
        else:
            scaled_len = int(round(width * (total / max_total)))
            scaled_len = max(0, min(width, scaled_len))

        if n_params == 0 or scaled_len == 0:
            bar = " " * width
            _log(f"bucket {b_id:2d} |{bar}| {total} ({_human(total)})  n={n_params}")
            continue

        slice_pos = buckets_slices_pos[b_id]
        if len(slice_pos) != dp_size + 1:
            bar = "[Invalid Slicing Info]" + " " * (width - 25)
            _log(f"bucket {b_id:2d} |{bar}| {total} ({_human(total)})  n={n_params}")
            continue

        rank_segment_sizes = [slice_pos[r+1] - slice_pos[r] for r in range(dp_size)]
        rank_char_counts = _alloc_proportional(scaled_len, rank_segment_sizes)
        
        pieces = []
        for r in range(dp_size):
            rank_char = str(r % 10)
            pieces.append(rank_char * rank_char_counts[r])
        
        segmented_bar = "".join(pieces)
        bar = segmented_bar + " " * (width - len(segmented_bar))

        _log(f"bucket {b_id:2d} |{bar}| {total} ({_human(total)})  n={n_params}")

def log_rank_bars(loads: List[int], title: str = "", width: int = 48) -> None:
    if title:
        _log(f"\n== {title} ==")
    obj = compute_objectives(loads)
    br = get_balance_ratio(loads)
    mx = max(loads) if loads else 0
    mu = obj["mean"]
    _log(f"total={_human(int(obj['total']))}  mean={_human(int(mu))}  "
          f"max={_human(int(mx))}  balance_ratio={br:.4f}  variance={obj['variance']:.2f}")
    for r, L in enumerate(loads):
        _log(f"rank {r:2d} |{_bar(L, mx, width=width)}| {L} ({_human(L)})")

def visualize_stage(
    buckets_params_range_map,
    buckets_slices_pos,
    dp_size: int,
    stage: str,
):
    _log("\n" + "=" * 80)
    _log(f"[{stage}]")
    _log("=" * 80)

    log_bucket_slice_bars(buckets_params_range_map, buckets_slices_pos, dp_size, title=f"{stage}: Bucket DP Slicing")

    cost_fn_mem = get_balance_cost_fn("numel")
    cost_fn_flops = get_balance_cost_fn("flops")
    loads_mem = visual_global_rank_loads(buckets_params_range_map, buckets_slices_pos, dp_size, cost_fn_mem)
    log_rank_bars(loads_mem, title=f"{stage}: DP Rank Memory Load")
    loads_flops = visual_global_rank_loads(buckets_params_range_map, buckets_slices_pos, dp_size, cost_fn_flops)
    log_rank_bars(loads_flops, title=f"{stage}: DP Rank FLOPs Load")



# =================================
# ------ Apply to Megatron -------
# =================================
from .distrib_optimizer import DistMatrixBasedOptimizer

def build_dp_load_balanced_dist_opt_buffer_slices_pos(
    buffer,
    dp_size,
):
    args = get_args()
    balance_cost_name = args.dp_balanced_opt_cost
    log_visualization = args.dp_balanced_opt_log_visualization
    dp_balanced_opt_alpha = args.dp_balanced_opt_alpha

    global dp_balance_cost
    dp_balance_cost = get_balance_cost_fn(balance_cost_name)

    buckets_params_range_map = []
    buckets_dp_slices_pos = []
    for bucket_index, _ in enumerate(buffer.buckets):  # per bucket in buckets
        _, param_to_in_bucket_range_map, bucket_dp_slices_pos = DistMatrixBasedOptimizer._build_model_gbuf_range(
            buffer, 
            bucket_index, 
            use_matrix_based_optim=True, 
            buffer_dp_slices_pos=None,
            dp_balance=True
        )
        buckets_params_range_map.append(param_to_in_bucket_range_map)
        buckets_dp_slices_pos.append(bucket_dp_slices_pos)

    # Original slicing rule
    loads = global_rank_loads(buckets_params_range_map, buckets_dp_slices_pos, dp_size)
    buckets_balance_ratio = get_balance_ratio(loads)
    if log_visualization:
        visualize_stage(
            buckets_params_range_map=buckets_params_range_map,
            buckets_slices_pos=buckets_dp_slices_pos,
            dp_size=dp_size,
            stage=f"BEFORE Greedy LPT",
        )
        for i, _ in enumerate(buckets_dp_slices_pos):
            print(f"bucket {i}'s slices position: {buckets_dp_slices_pos[i]}")
        print(f"dp balance ratio: {get_balance_ratio(loads):.6}")
        for bucket_idx, bsp in enumerate(buckets_dp_slices_pos):
            print(f"bucket {bucket_idx} comm balance ratio: "
                f"{get_balance_ratio([bsp[r+1] - bsp[r] for r in range(dp_size)]):.6}")

    real_bucket_totals = [slices[-1] for slices in buckets_dp_slices_pos]

    # Bucket-Constrained Iterative Balancing inside each 'param_and_grad_buffer'
    new_buckets_dp_slices_pos = greedy_lpt_with_ranges(
        buckets_params_range_map=buckets_params_range_map,
        dp_size=dp_size,
        explicit_bucket_totals=real_bucket_totals,
        bucket_order="by_total_desc",
        alpha=dp_balanced_opt_alpha
    )
    loads = global_rank_loads(buckets_params_range_map, new_buckets_dp_slices_pos, dp_size)
    new_buckets_balance_ratio = get_balance_ratio(loads)
    if log_visualization:
        visualize_stage(
            buckets_params_range_map=buckets_params_range_map,
            buckets_slices_pos=new_buckets_dp_slices_pos,
            dp_size=dp_size,
            stage=f"AFTER Greedy LPT",
        )
        for i, _ in enumerate(new_buckets_dp_slices_pos):
            print(f"new bucket {i}'s slices position: {new_buckets_dp_slices_pos[i]}")
        print(f"dp balance ratio: {get_balance_ratio(loads):.6}")
        for bucket_idx, bsp in enumerate(new_buckets_dp_slices_pos):
            print(f"bucket {bucket_idx} comm balance ratio: "
                f"{get_balance_ratio([bsp[r+1] - bsp[r] for r in range(dp_size)]):.6}")

    # Stick with the original plan if the new plan isn't strictly better
    if new_buckets_balance_ratio >= buckets_balance_ratio or int(os.environ.get('DEBUG_DP_BALANCED_OPT', 0)) == 1:
        new_buckets_dp_slices_pos = buckets_dp_slices_pos

    return new_buckets_dp_slices_pos
