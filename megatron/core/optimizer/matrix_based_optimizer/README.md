# Canzona: Matrix-Based Optimizers in Megatron

> **Paper:** [Canzona: Unified, Asynchronous, and Load-Balanced Matrix-Based Optimization for Large-Scale Distributed Training](https://arxiv.org/abs/2602.06079)

## Overview

Canzona enables matrix-based optimizers (such as **Muon** and **SOAP**) to run efficiently within Megatron-LM's distributed training framework. The core challenge it solves is the fundamental conflict between these optimizers and Megatron's parallelism:

- **Matrix-based optimizers** require complete (non-sharded) weight matrices to compute preconditioners or orthogonalization (e.g., Newton-Schulz iteration in Muon, eigen-decomposition in SOAP).
- **Megatron's Tensor Parallelism (TP)** splits weight matrices across GPUs, fragmenting the data the optimizer needs.

Existing workarounds suffer from either computational redundancy (synchronous gathering) or load imbalance (naive per-layer partitioning). Canzona resolves this by **decoupling logical optimizer assignment from physical parameter distribution**.

## Results

Evaluated on Qwen3 models (up to 32B parameters) on 256 GPUs:
- **1.57x speedup** in end-to-end iteration time
- **5.8x reduction** in optimizer step latency vs. baseline

## Architecture

```
                    Canzona Module Structure
   ┌─────────────────────────────────────────────────────────┐
   │                    Megatron Training                    │
   │  ┌─────────────────────────────────────────────────┐    │
   │  │                 ChainedOptimizer                │    │
   │  │  ┌─────────────────┐ ┌───────────────────────┐  │    │
   │  │  │  Dense Optim    │ │ MoE Expert-Parallel   │  │    │
   │  │  │  (Adam + Muon/  │ │  (Muon/SOAP + Adam)   │  │    │
   │  │  │  SOAP mixed)    │ │                       │  │    │
   │  │  └────────┬────────┘ └───────────┬───────────┘  │    │
   │  └───────────┼──────────────────────┼──────────────┘    │
   │              │                      │                   │
   │  ┌───────────▼──────────────────────▼───────────────┐   │
   │  │           DistMatrixBasedOptimizer               │   │
   │  │  ┌────────────────────────┐ ┌─────────────────┐  │   │
   │  │  │ DP: Load-Balanced      │ │ TP: Async       │  │   │
   │  │  │   Partitioning         │ │   Micro-Group   │  │   │
   │  │  │ (greedy LPT + alpha)   │ │   Scheduling    │  │   │
   │  │  └────────────────────────┘ └─────────────────┘  │   │
   │  └──────────────────────┬───────────────────────────┘   │
   │                         │                               │
   │  ┌──────────────────────▼───────────────────────────┐   │
   │  │            Matrix-Based Optimizers               │   │
   │  │  ┌──────────────┐          ┌──────────────────┐  │   │
   │  │  │  Muon        │          │  SOAP            │  │   │
   │  │  │ (NS iters +  │          │ (Adam + Shampoo  │  │   │
   │  │  │  momentum)   │          │  preconditioner) │  │   │
   │  │  └──────────────┘          └──────────────────┘  │   │
   │  └──────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────┘
```

### Code Organization

```
megatron/core/optimizer/matrix_based_optimizer/
├── __init__.py                              # Public exports, param group wiring
├── distrib_optimizer.py                     # DistMatrixBasedOptimizer: core distributed optimizer
├── param_and_grad_buffer.py                 # _MatrixBasedParamAndGradBucketGroup: bucket management
├── load_balanced_dp_buffer.py               # DP load-balancing (greedy LPT + alpha)
├── load_balanced_tp_executor.py             # TP async execution (AsyncGroupExecutor)
├── split_grad_and_state.py                  # GradAndStateSplitter: parameter splitting
├── comm_extension/                          # Coalesced all-gather-v / reduce-scatter-v primitives
├── utils.py                                 # FLOPs estimation, tagging predicates
├── optimizers/
│   ├── README.md                            # Adding a new matrix-based optimizer
│   ├── base_optim.py                        # BaseOptim: abstract base class
│   ├── muon.py                              # Muon (Newton-Schulz orthogonalization)
│   └── soap.py                              # SOAP (Shampoo + Adam)
└── README.md                                # This file
```

## Key Design Components

### 1. Mixed Optimizer Strategy

Canzona does not replace all optimizers with matrix-based ones. Instead, it intelligently assigns optimizers based on parameter type:

- **Matrix-based (Muon/SOAP):** 2D weight matrices (e.g., `linear_qkv.weight`, `linear_fc1.weight`) that benefit from second-order information. Embeddings (`word_embeddings`), output layers (`output_layer`), and MoE routers (`router`, `gate_weight`) are excluded.
- **Adam:** All other parameters (embeddings, 1D biases, MoE router weights, etc.).

This is achieved by tagging parameters in `_get_param_groups()` based on the `is_param_use_matrix_based_optim()` predicate from [`utils.py`](utils.py). A parameter qualifies for matrix-based optimization if:

1. It is exactly 2-dimensional (`param.ndim == 2`), matching the shape requirement for Newton-Schulz iteration or eigen-decomposition.
2. Its name does **not** contain any of the following substrings:
   - `word_embeddings` — embedding matrices are typically very large and sparse, making matrix-based methods less effective.
   - `output_layer` — the final projection layer maps hidden states to vocabulary (often 100K+), where matrix-based optimization is prohibitively expensive.
   - `router` / `gate_weight` — MoE routing weights are lightweight parameters that don't benefit from second-order methods.

The optimizer then creates separate param groups with `use_muon` or `use_soap` flags, while all remaining parameters fall into the default Adam group.

### 2. Data Parallelism: Load-Balanced Partitioning (`load_balanced_dp_buffer.py`)

In Megatron's Distributed Optimizer (ZeRO-1), gradients are reduce-scattered and params are all-gathered across DP ranks. When using matrix-based optimizers, per-rank computation can become imbalanced because parameters are not evenly divisible by their optimization cost.

**Solution:** Alpha-Balanced Static Partitioning

```
┌─────────────────────────────────────────┐
│  Bucket (total gradient buffer)         │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐      │
│  │ 0 │ 1 │ 2 │ 0 │ 2 │ 1 │ 0 │ 1 │      │  DP ranks
│  └───┴───┴───┴───┴───┴───┴───┴───┘      │
│                                         │
│  Greedy LPT: each bucket sliced at      │
│  param boundaries to balance load       │
│  across ranks while keeping params      │
│  atomic (no param split across ranks)   │
└─────────────────────────────────────────┘
```

The `greedy_lpt_with_ranges()` algorithm:
1. Sorts buckets by total size (descending)
2. For each bucket, computes target allocations blending perfect DP balance (`alpha=1.0`) and perfect comm balance (`alpha=0.0`)
3. Discretizes allocations to param boundaries (preserving param atomicity)
4. Falls back to original sharding if the new plan is not better

**Cost models:** `--dp-balanced-opt-cost numel` (default, by parameter count) or `flops` (estimated optimizer FLOPs).

### 3. Tensor Parallelism: Async Compute Pipeline (`load_balanced_tp_executor.py`)

For TP-sharded parameters, the optimizer must gather full gradients before computing the update. Canzona's `AsyncGroupExecutor` implements a Gather → Compute → Scatter → Update pipeline:

```
Rank 0:  [Gather grads] → [Compute update] → [Scatter updates] → [Apply update]
Rank 1:  [Send grads]   →    (idle)        → [Receive updates] → [Apply update]
Rank 2:  [Send grads]   →    (idle)        → [Receive updates] → [Apply update]
```

To avoid stragglers, Canzona introduces **Micro-Group Scheduling**: TP params are organized into micro-groups where each group distributes work across TP ranks in a load-balanced manner. Four scheduling modes:

| Mode | Description |
|------|-------------|
| `no` | Simple micro-grouping without load balancing |
| `single` | Per-micro-group balance (sort params, distribute round-robin) |
| `slot` | Heap-based scheduling with per-slot capacity limits (`--tp-balanced-opt-fuse-space`) |
| `global` | Globally optimal scheduling (partition solve with cumulative cost tracking) |

Communication can be fused via `all_to_all_single` (`--no-async-tp-fuse-comm` to disable) or done per-parameter with `gather`/`scatter`.

### 4. Parameter Splitting (`split_grad_and_state.py`)

Certain large weight matrices can be further split into sub-matrices for finer-grained optimization. This is controlled by flags and parameter attributes:

| Flag | Target Param | Split Method | Granularity |
|------|-------------|--------------|-------------|
| `--matrix-based-optimizer-split-qkv` | `linear_qkv.weight` | `is_full_attn_qkv` | Q/K/V or per-head |
| `--matrix-based-optimizer-split-fc1` | `linear_fc1.weight` | `is_mlp_fc1`, `is_expert_fc1`, `is_shared_expert_fc1` | Gate/Up branches |
| `--matrix-based-optimizer-split-linear-attn` | `in_proj.weight` | `is_linear_attn_inproj` | Q/K/V/Z/B/A or per-head |

The `GradAndStateSplitter` handles splitting gradients into 2D sub-fragments before the optimizer step and reassembling them afterward, maintaining separate optimizer state for each fragment.

### 5. Adaptive Bucket Sizing (`param_and_grad_buffer.py`, `comm_extension/`)

Canzona adjusts bucket boundaries so each DP rank receives an equal share of parameters, maximizing **even buckets** — where every DP rank gets an identical shard size. Even buckets use fast native PyTorch collectives (`all_gather_into_tensor`, `reduce_scatter_tensor`), while remaining **uneven buckets** fall back to coalesced custom `all-gather-v` / `reduce-scatter-v` primitives from [`comm_extension/`](./comm_extension/).

The `_MatrixBasedParamAndGradBucketGroup._classify_buckets()` method checks whether all shards within a bucket have equal size:
```python
even_shard = len(set([r.end - r.start for r in bucket.real_gbuf_world_ranges])) == 1
```

Buckets are classified into `even_buckets` and `uneven_buckets`, allowing different communication paths to run in parallel via `torch.distributed._coalescing_manager`. Bucket size is controlled via environment variables `MATRIX_BASED_OPTIM_DENSE_BUCKET_SIZE` and `MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE`.

### 6. Distributed Checkpointing

`DistMatrixBasedOptimizer` implements `sharded_state_dict()` with `fully_sharded_model_space` sharding. Optimizer states (momentum buffers, preconditioner matrices, etc.) are saved per-param-shard and can be reloaded at different DP/TP configurations. The `sharded_param_state_fs_model_space()` method handles the mapping between model param shards and their optimizer states, including special handling for TP-sharded parameters.


## Supported Optimizers

### Muon (MomentUM Orthogonalized by Newton-Schulz)

Applies SGD-momentum followed by Newton-Schulz iteration to compute the nearest orthogonal matrix of the gradient.

- **Core operation:** `zeropower_via_newtonschulz5()` — a quintic polynomial iteration in BF16
- **Learning rate scaling:** `lr * 0.2 * sqrt(max(A, B))` where A, B are matrix dimensions
- **Coefficient sets:** `simple`, `quintic`, `polar_express`, `aol_nvidia`, `qwen_express`
- **State:** Per-parameter `momentum_buffer`

### SOAP (Scalable Second-Order Preconditioner)

Combines Adam with Shampoo-style preconditioning via eigenvalue decomposition of gradient outer products.

- **Hot path:** Project gradients through eigenbases Q_L, Q_R of GG matrices, apply Adam update
- **Cold path:** Periodically update preconditioner via eigen-decomposition (`--soap-precondition-frequency`)
- **State:** `exp_avg`, `exp_avg_sq`, `GG_0`, `GG_1`, `Q_0`, `Q_1`, `step`

### Adding a New Matrix-Based Optimizer

- This README — architecture overview, usage guide, and configuration reference for the Canzona framework.
- **[optimizers/README.md](optimizers/README.md)** — step-by-step guide for adding a new matrix-based optimizer. Covers the required changes across 5 files (optimizer class, utils tagging, checkpointing, optimizer selection wiring, and config/CLI).

## Usage

### Basic Configuration

```bash
# Enable Muon optimizer
--optimizer muon

# Or enable SOAP optimizer
--optimizer soap
```

### DP & TP Adaptation

```bash
# DP load balancing (requires --use-distributed-optimizer, --overlap-grad-reduce, --overlap-param-gather)
--use-dp-balanced-opt
--dp-balanced-opt-alpha 1.0          # 1.0 = pure DP balance, 0.0 = pure comm balance
--dp-balanced-opt-cost flops         # "numel" or "flops"
--dp-balanced-opt-log-visualization  # Output dp-balance visualization to log

# TP async (enabled by default for matrix-based optimizers)
--use-tp-sync-opt                   # Use this flag to DISABLE async TP
--no-async-tp-fuse-comm             # Disable fused all-to-all communication

# TP load-balanced micro-group scheduling
--use-tp-balanced-opt               # Enable TP load-balanced scheduling
--tp-balanced-opt-cost flops        # "numel" or "flops"
--tp-balanced-opt-fuse-space 400    # Max slot size in MB
--tp-balanced-opt-log-visualization # Output tp-balance visualization to log
```

### CUDA Graph

```bash
export USE_CUDA_GRAPH_OPTIM=1         # Enable CUDA graph for optimizer compute
```

### Adaptive DP bucket sizing
See [bucket_size_calculator.py](../../../../scripts/canzona/bucket_size_calculator.py)
```bash
# Enable higher performance communication operators
export MATRIX_BASED_OPTIM_DENSE_BUCKET_SIZE=400000000
export MATRIX_BASED_OPTIM_EXPERT_BUCKET_SIZE=400000000
```

### Parameter Splitting

```bash
# Split QKV weights into per-head matrices for Muon/SOAP
--matrix-based-optimizer-split-qkv
--matrix-based-optimizer-split-qkv-per-head  # Finer per-head granularity

# Split FC1 (SwiGLU) into gate/up branches
--matrix-based-optimizer-split-fc1

# Split linear attention in-projection
--matrix-based-optimizer-split-linear-attn
--matrix-based-optimizer-split-linear-attn-per-head
```

### Muon-Specific Options

```bash
--muon-ns-steps 5                   # Newton-Schulz iteration steps
--muon-ns-coefficient-type simple   # "simple" | "quintic" | "polar_express" | "aol_nvidia" | "qwen_express"
--muon-ns-norm-eps 1e-7             # Normalization epsilon
--nesterov-acceleration             # Enable Nesterov momentum
```

### SOAP-Specific Options

```bash
--soap-precondition-frequency 10    # How often to update Q matrices
--soap-max-precond-dim 10000        # Max dimension for preconditioner
--shampoo-beta 0.99                 # Beta for GG moving average
--soap-correct-bias                 # Bias correction
```

## Example Scripts

See [`scripts/canzona/README.md`](../../../../scripts/canzona/README.md) for reference training scripts:
- `scripts/canzona/prepare.sh` — Environment preparation
- `scripts/canzona/train.sh` — Training launch script

## Roadmap

- **HSDP (Hybrid Sharded Data Parallel)** — In scenarios with large DP size and small model parameter count, the number of load-balance "chunks" is insufficient — too few parameters to distribute evenly across many DP ranks. HSDP extends DP load-balancing to hybrid sharding topologies (DP × FSDP), enabling finer-grained partitioning by treating per-shard parameters as the balancing unit rather than full-model parameters, thereby improving load balance in wide-but-shallow configurations.
- **Parameter-Splitting-Aware Load Balancing** — The "Parameter Splitting" feature already breaks large weight matrices (e.g., QKV, FC1) into smaller sub-matrices. Currently, DP load-balancing operates at the original parameter granularity, leaving the extra granularity from splitting on the table. Future work will propagate split-level information to the load-balancer so that sub-parameters can be independently assigned to DP ranks, enabling finer-grained scheduling and better load balance.
- **More Optimizers** — the plugin API (`optimizers/base_opt.py`) defines a minimal interface that matrix-based optimizers must implement. This enables third-party or research optimizers to plug into Canzona without modifying the distributed optimizer core. Planned integrations include SSO and other second-order methods.
- **Higher-Performance Communication Primitives** — uneven buckets (where DP ranks receive shards of different sizes) are currently handled by coalesced custom `all-gather-v` / `reduce-scatter-v` operations, which satisfy the functional requirements. There is still room for further optimization through more efficient kernel scheduling.
- **Checkpointing** — `DistMatrixBasedOptimizer` currently only supports `fully_sharded_model_space` sharding for checkpoint save/restore. Future work will add support for `dp_reshardable` and `fully_reshardable` sharding modes, enabling checkpoint restoration across different DP/TP/PP configurations without requiring manual state resharding. This is critical for elastic training where cluster topology may differ between runs.
