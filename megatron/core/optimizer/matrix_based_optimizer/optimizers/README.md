# Adding a New Matrix-Based Optimizer

This guide explains how to integrate a new matrix-based optimizer (e.g., a new preconditioned optimizer, a new orthogonalization-based optimizer) into the Canzona framework.

## Overview

Adding a new optimizer requires changes in **5 locations** across the codebase:

```
1. optimizers/your_optimizer.py          ← New optimizer class
2. matrix_based_optimizer/utils.py       ← Tagging predicates
3. matrix_based_optimizer/distrib_optimizer.py  ← Checkpointing (state allocation)
4. optimizer/__init__.py                 ← Optimizer selection & param group wiring
5. optimizer_config.py + arguments.py    ← Configuration & CLI
```

Below is a step-by-step walkthrough using a hypothetical optimizer `MyOptim` as an example.

---

## Step 1: Implement the Optimizer Class

**File:** `megatron/core/optimizer/matrix_based_optimizer/optimizers/my_optimizer.py`

Create a new class that inherits from `BaseOptim`. You must implement `_inner_single_param_step` and `_single_param_update`:

```python
import torch
from .base_optim import BaseOptim

class MyOptim(BaseOptim):
    """
    MyOptim - Your optimizer description.

    Usage:
        opt = MyOptim(
            params=[{"params": my_optim_params, "use_my_optim": True},
                    {"params": adamw_params, "use_my_optim": False}],
            lr=1e-3,
            ...
        )
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        # ... your hyperparams ...
        split_my_optim_params=False,
        split_my_optim_shape_map=None,
        async_tp=False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            # ... your hyperparams ...
        )
        super().__init__(params, defaults, split_my_optim_params, split_my_optim_shape_map, async_tp)

    def _inner_single_param_step(self, name, p, grad, group):
        """
        Compute the update tensor for a single parameter.

        This method is called per-parameter (or per-split-fragment if parameter
        splitting is enabled). It should:
        1. Initialize optimizer state on the first call (e.g., momentum buffers,
           preconditioner matrices) using `state.setdefault()`.
        2. Compute the update tensor based on the gradient and optimizer state.
        3. Return the update tensor.

        Args:
            name: Prefix string for state keys (empty for non-split params,
                  e.g. "is_mlp_fc1.0." for split fragments).
            p: The parameter tensor.
            grad: The gradient tensor (full matrix for non-TP, gathered gradient
                  for TP-sharded parameters).
            group: The parameter group dict containing hyperparameters.

        Returns:
            The update tensor (same shape as grad). Return None to skip the update.
        """
        # Example:
        state = self.state[p]
        buf = state.setdefault(f"{name}my_buffer", torch.zeros_like(grad))
        # ... compute update ...
        return update

    def _single_param_update(self, p, u, group):
        """
        Apply the update tensor `u` to parameter `p` in-place.

        This method performs the actual parameter update (e.g., p.data.add_()).
        It is called after `_inner_single_param_step` returns the update tensor.

        Args:
            p: The parameter tensor.
            u: The update tensor computed by `_inner_single_param_step`.
            group: The parameter group dict.
        """
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        p.data.mul_(1 - lr * weight_decay)
        p.data.add_(-lr * u)
```

**Key design points:**
- `BaseOptim.step()` handles the orchestration: iterating over param groups, dispatching TP params to `AsyncGroupExecutor`, and managing split parameters via `GradAndStateSplitter`.
- If your optimizer needs per-head or per-branch splitting (like Muon/SOAP for QKV weights), pass `split_my_optim_params=True` and `split_my_optim_shape_map=...` to the parent constructor. The `GradAndStateSplitter` will automatically split gradients before `_inner_single_param_step` and reassemble updates afterward.
- For TP-sharded params, the `AsyncGroupExecutor` gathers full gradients, calls `_single_param_step`, scatters updates back, and applies them via `_single_param_update`. No extra TP handling is needed in your optimizer class.

> **CUDA Graph note:** Enabling CUDA graph for a new optimizer (`USE_CUDA_GRAPH_OPTIM=1`) requires writing a dedicated graph-capture function for your optimizer's compute kernel (see `optimizers/muon.py` and `optimizers/soap.py` for reference). This involves non-trivial engineering effort to ensure correct state management, split-parameter handling, and TP synchronization within the captured graph.

---

## Step 2: Register Tagging Predicates

**File:** `megatron/core/optimizer/matrix_based_optimizer/utils.py`

Add the tag to `is_matrix_based_optim()` so the framework recognizes your optimizer:

```python
def is_matrix_based_optim(optim_name=None):
    from megatron.training import get_args
    args = get_args()
    optim_name = args.optimizer if optim_name is None else optim_name
    if (
        optim_name == "muon"
        or optim_name == "soap"
        or optim_name == "my_optim"     # <-- Add your optimizer
    ):
        return True
    else:
        return False
```

Add the group-level predicate to `is_matrix_based_optim_group()`:

```python
def is_matrix_based_optim_group(param_group):
    if (
        param_group['use_muon']
        or param_group['use_soap']
        or param_group['use_my_optim']    # <-- Add your optimizer
    ):
        return True
    else:
        return False
```

Optionally, update `is_param_use_matrix_based_optim()` if your optimizer has different criteria for selecting which parameters to apply it to:

```python
def is_param_use_my_optim(name: str, param: torch.Tensor) -> bool:
    from megatron.training import get_args
    args = get_args()
    return (
        param.ndim == 2
        and 'word_embeddings' not in name
        and 'output_layer' not in name
        # ... your selection logic ...
    )
```

Export the new function from `matrix_based_optimizer/__init__.py`:

```python
from .utils import is_param_use_my_optim
```

---

## Step 3: Add Checkpointing Support

**File:** `megatron/core/optimizer/matrix_based_optimizer/distrib_optimizer.py`

Two places in `DistMatrixBasedOptimizer` need updates:

### 3a. Optimizer identification in `__init__`

Add your optimizer to the `self.use_optimizer` detection:

```python
if isinstance(optimizer, Adam):
    self.use_optimizer = 'adam'
elif isinstance(optimizer, Muon):
    self.use_optimizer = 'muon'
elif isinstance(optimizer, SOAP):
    self.use_optimizer = 'soap'
elif isinstance(optimizer, MyOptim):     # <-- Add
    self.use_optimizer = 'my_optim'      # <-- Add
else:
    raise NotImplementedError(f"Unsupported optimizer: {type(optimizer)}")
```

Also add the import at the top of the file:

```python
from .optimizers import MyOptim
```

### 3b. State allocation in `load_state_dict()`

In the state allocation block (around line 880), add a branch for your optimizer's state tensors. This ensures dummy state is allocated during checkpoint loading:

```python
if self.use_optimizer == 'adam':
    tensors = {
        "exp_avg": init_shard(self.config.exp_avg_dtype),
        "exp_avg_sq": init_shard(self.config.exp_avg_sq_dtype),
    }
elif self.use_optimizer == 'muon':
    tensors = {"momentum_buffer": init_shard()}
elif self.use_optimizer == 'soap':
    soap_state_dict = {"exp_avg": init_shard(), "exp_avg_sq": init_shard()}
    # ... GG, Q, step allocation ...
    tensors = soap_state_dict
elif self.use_optimizer == 'my_optim':    # <-- Add
    tensors = {                            # <-- Add
        "my_buffer": init_shard(),          # <-- Add your state keys
    }                                       # <-- Add
```

If your optimizer supports split parameters, also handle the split-state case (similar to the existing `split_methods` branches for Muon and SOAP).

---

## Step 4: Wire Up Optimizer Selection

**File:** `megatron/core/optimizer/__init__.py`

### 4a. Import the optimizer

```python
from .matrix_based_optimizer import MyOptim
```

### 4b. Tag parameters in `_get_param_groups()`

Add your optimizer's flag to the parameter grouping key:

```python
use_my_optim = is_param_use_my_optim(name, param) if args.optimizer == 'my_optim' else False

key = (wd_mult, _lr_mult, is_expert_parallel, is_tensor_parallel, is_decoupled_lr,
       use_muon, use_soap, use_my_optim)   # <-- Add to key tuple
```

And to the param group dict:

```python
param_group = {
    'params': params,
    ...
    'use_muon': use_muon,
    'use_soap': use_soap,
    'use_my_optim': use_my_optim,          # <-- Add
}
```

Update the assertion for param_group_identifier_keys:

```python
assert set(param_group.keys()) - set(param_group_identifier_keys) == {
    'params',
    'is_tensor_parallel',
    'use_muon',
    'use_soap',
    'use_my_optim',                         # <-- Add
}
```

### 4c. Instantiate the optimizer in `_get_megatron_optimizer_based_on_param_groups()`

Add the construction branch:

```python
elif config.optimizer == 'my_optim' and param_groups[0]['use_my_optim']:
    optimizer = MyOptim(
        param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
        # ... your hyperparams from config ...
        split_my_optim_params=config.split_matrix_based_optimizer_params,
        split_my_optim_shape_map=split_my_optim_shape_map,
        async_tp=config.use_tp_async_opt,
    )
    def init_state_fn(opt):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]["my_buffer"] = torch.zeros_like(p.data)
```

### 4d. Handle mixed optimizer groups

If some params use your optimizer and others use Adam, ensure the fallback branch handles it:

```python
elif config.optimizer == 'adam' or (is_matrix_based_optim(config.optimizer) and not is_matrix_based_optim_group(param_groups[0])):
    # This branch handles Adam for non-matrix-based params
```

This condition is already correct — it falls through to Adam when the current group is not tagged for your optimizer.

---

## Step 5: Add Configuration

### 5a. Config dataclass

**File:** `megatron/core/optimizer/optimizer_config.py`

Add your optimizer's config fields to `OptimizerConfig`:

```python
################
# MyOptim optimizer
################
my_optim_hyperparam1: float = 0.1
"""Description of hyperparam1."""

my_optim_hyperparam2: int = 5
"""Description of hyperparam2."""
```

### 5b. CLI arguments

**File:** `megatron/training/arguments.py`

Add CLI arguments in `_add_canzona_args()`:

```python
# MyOptim specific arguments
group.add_argument('--my-optim-hyperparam1', type=float, default=0.1,
                   help='Description of hyperparam1.')
group.add_argument('--my-optim-hyperparam2', type=int, default=5,
                   help='Description of hyperparam2.')
```

### 5c. Validation (optional)

In `validate_args()`, add any constraints:

```python
if args.optimizer == 'my_optim':
    assert not args.use_megatron_fsdp, "MyOptim does not support FSDP."
    assert not args.optimizer_cpu_offload, "MyOptim does not support CPU offload."
```

### 5d. Parameter splitting (optional)

If your optimizer needs custom parameter splitting (like QKV/FC1 splitting), add:

1. A flag in `arguments.py`:
   ```python
   group.add_argument('--matrix-based-optimizer-split-xxx', action='store_true')
   ```

2. A split attribute in `split_grad_and_state.py`:
   ```python
   SPLIT_ATTRS = [
       'is_full_attn_qkv',
       ...
       'is_my_split_method',             # <-- Add
   ]
   ```

3. Tagging logic in `_get_param_groups()` in `__init__.py`:
   ```python
   if args.matrix_based_optimizer_split_xxx and 'xxx.weight' in name:
       param.is_my_split_method = True
   ```

4. Shape map computation in `_get_megatron_optimizer_based_on_param_groups()`:
   ```python
   if args.matrix_based_optimizer_split_xxx:
       split_my_optim_shape_map['is_my_split_method'] = [...]
   ```

5. Split/gather logic in `GradAndStateSplitter.split()` and `gather()`.

---

## Checklist

- [ ] New optimizer class in `megatron/core/optimizer/matrix_based_optimizer/optimizers/my_optimizer.py` inherits `BaseOptim`
- [ ] `_inner_single_param_step()` and `_single_param_update()` implemented
- [ ] `is_matrix_based_optim()` updated in `megatron/core/optimizer/matrix_based_optimizer/utils.py`
- [ ] `is_matrix_based_optim_group()` updated in `megatron/core/optimizer/matrix_based_optimizer/utils.py`
- [ ] Import and `use_optimizer` detection in `megatron/core/optimizer/matrix_based_optimizer/distrib_optimizer.py`
- [ ] State allocation in `load_state_dict()` in `megatron/core/optimizer/matrix_based_optimizer/distrib_optimizer.py`
- [ ] Import in `megatron/core/optimizer/__init__.py`
- [ ] Param group key + dict extended in `_get_param_groups()` in `megatron/core/optimizer/__init__.py`
- [ ] Optimizer instantiation in `_get_megatron_optimizer_based_on_param_groups()` in `megatron/core/optimizer/__init__.py`
- [ ] Config fields in `megatron/core/optimizer/optimizer_config.py`
- [ ] CLI arguments in `megatron/training/arguments.py`
- [ ] Validation rules in `validate_args()` in `megatron/training/arguments.py`
- [ ] Import in `megatron/core/optimizer/matrix_based_optimizer/__init__.py`
