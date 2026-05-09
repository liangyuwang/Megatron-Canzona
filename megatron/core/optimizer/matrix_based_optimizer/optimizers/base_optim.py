import torch

from megatron.core.optimizer.matrix_based_optimizer.load_balanced_tp_executor import AsyncGroupExecutor, SyncGroupExecutor, is_group_tensor_parallel
from megatron.core.optimizer.matrix_based_optimizer.split_grad_and_state import GradAndStateSplitter
import gc
import os
import copy

class BaseOptim(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        defaults,
        split_params = False,
        split_shape_map = None,
        async_tp = False,
    ):
        # Let base class normalize groups and fill defaults
        super().__init__(params, defaults)
        self.grad_and_state_splitter = GradAndStateSplitter(split_params, split_shape_map)
        if async_tp:
            self.tp_param_group_executor = AsyncGroupExecutor()
        else:
            self.tp_param_group_executor = SyncGroupExecutor()
        # add cuda graph
        self.use_cuda_graph = int(os.environ.get('USE_CUDA_GRAPH_OPTIM', 0)) == 1
        if self.use_cuda_graph:
            self._cuda_graphs = {}
            self._cuda_graph_warmup_steps = 3
            self._cuda_graph_current_step = 0
            self._group_meta = copy.deepcopy(defaults)
            self._mempool = torch.cuda.graph_pool_handle()
    
    def _single_param_update(self, p, u, group):
        dist_optim = 'origin_shape' in group
        if dist_optim:
            u = u.reshape(-1)
        p.data.add_(u)

    def _single_param_step(self, p, s, group, g=None):
        if is_group_tensor_parallel(group) and g is None:
            raise RuntimeError(f"'g' must be provided when using TP.")
        g = p.grad.view(s) if g is None else g

        true_attrs = self.grad_and_state_splitter.get_split_param_methods(p)
        if true_attrs:
            assert len(true_attrs) == 1, f"Only one of {self.grad_and_state_splitter.get_attrs()} can be set for a param, got {true_attrs}"
            split_method = true_attrs[0]
            grads = self.grad_and_state_splitter.split(g, split_method, g.shape)  # use g.shape, not s
            grads = [self._inner_single_param_step(f"{split_method}.{idx}.", p, gg, group)
                for idx, gg in enumerate(grads)]
            if None in grads:
                return torch.zeros_like(g)
            u = self.grad_and_state_splitter.gather(grads, split_method, g.shape)  # use g.shape, not s

        else:
            u = self._inner_single_param_step("", p, g, group)
            if u is None:
                return torch.zeros_like(g)
        return u.to(g.dtype)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self.use_cuda_graph:
            return self.step_with_cuda_graph(loss)
        
        for group in self.param_groups:
            # Increment step counter to stay consistent with TE FusedAdam.
            # _synchronize_steps() in ChainedOptimizer relies on this for
            # cross-optimizer step alignment.
            group["step"] = group.get("step", 0) + 1
            if 'origin_shape' not in group:
                shapes_map = {p: {'origin_shape': p.shape} for p in group['params']}
                shapes = [p.shape for p in group['params']] # compatible with no dist-optim
            else:
                shapes_map = {p: {'origin_shape': s}  for p, s in zip(group['params'], group["origin_shape"])}
                shapes = group["origin_shape"]
            group["shapes_map"] = shapes_map
            if not is_group_tensor_parallel(group):
                for p, s in zip(group["params"], shapes):
                    if p.grad is None:
                        continue
                    tensor_to_update_p = self._single_param_step(p, s, group)
                    self._single_param_update(p, tensor_to_update_p, group)
            else:
                self.tp_param_group_executor.execute(group, shapes, self._single_param_step, self._single_param_update)
            group.pop("updated", None)
        return loss

    def _inner_single_param_step(self, name, p, grad, group):
        """
        API for a single param's optimizer step
        """
        raise NotImplementedError("_inner_single_param_step must be implemented in subclass.")

    def _get_graph_specs(self):
        """Return the list of (name, kwargs) pairs describing graphs to capture, in order.

        Each entry causes one capture step. During replay, _select_graph_for_replay()
        picks which graph to use based on the current replay step counter.

        Default: a single graph with no extra kwargs (one-path optimizers like Muon).
        Override in subclasses for multi-path optimizers (e.g. SOAP uses two graphs).
        """
        return [('default', {})]

    def _select_graph_for_replay(self, group, graphs, replay_step):
        """Return the CUDAGraph to replay for this step.

        graphs: dict[name -> CUDAGraph] built by step_with_cuda_graph during capture.
        replay_step: 0-based counter that increments once per call after all captures.

        Default: always replay the single 'default' graph.
        Override in subclasses to implement step-dependent dispatch.
        """
        return graphs['default']

    def step_with_cuda_graph(self, loss):
        specs = self._get_graph_specs()          # [(name, kwargs), ...]
        num_captures = len(specs)

        for i, group in enumerate(self.param_groups):
            group["step"] = group.get("step", 0) + 1
            # Build shapes_map consistently with step() so subclasses have access to it
            if 'origin_shape' not in group:
                shapes_map = {p: {'origin_shape': p.shape} for p in group['params']}
                shapes = [p.shape for p in group['params']]
            else:
                shapes_map = {p: {'origin_shape': s}
                              for p, s in zip(group['params'], group["origin_shape"])}
                shapes = group["origin_shape"]
            group["shapes_map"] = shapes_map

            self._upload_group_meta_to_cuda_graph(group, shapes)

            if not is_group_tensor_parallel(group):
                current_step = self._cuda_graph_current_step

                if current_step < self._cuda_graph_warmup_steps:
                    # Run warmup on a dedicated stream so lazy CUDA inits (cuDNN benchmarking,
                    # memory allocator warm-up) are never recorded into the graph.
                    if current_step == 0:
                        torch.cuda.synchronize()  # flush all pending ops before warmup
                    warmup_stream = torch.cuda.Stream()
                    with torch.cuda.stream(warmup_stream):
                        self._run_param_updates(group, shapes)
                    if current_step == self._cuda_graph_warmup_steps - 1:
                        torch.cuda.synchronize()  # ensure warmup fully done before capture

                elif current_step < self._cuda_graph_warmup_steps + num_captures:
                    # Capture phase: one capture step per graph spec, in order.
                    # Disable GC to avoid PyTorch bug (pytorch/pytorch#161037).
                    capture_idx = current_step - self._cuda_graph_warmup_steps
                    name, kwargs = specs[capture_idx]
                    graph = torch.cuda.CUDAGraph()
                    gc_enabled = gc.isenabled()
                    if gc_enabled:
                        gc.disable()
                    with torch.cuda.graph(graph, pool=self._mempool):
                        self._run_param_updates(group, shapes, **kwargs)
                    if gc_enabled:
                        gc.enable()
                    torch.cuda.synchronize()
                    if i not in self._cuda_graphs:
                        self._cuda_graphs[i] = {}
                    self._cuda_graphs[i][name] = graph

                else:
                    # Replay: delegate graph selection to the subclass hook.
                    replay_step = current_step - self._cuda_graph_warmup_steps - num_captures
                    graph = self._select_graph_for_replay(group, self._cuda_graphs[i], replay_step)
                    graph.replay()
            else:
                self.tp_param_group_executor.execute(
                    group, shapes,
                    self._single_param_step,
                    self._single_param_update
                )

            self._offload_group_meta_from_cuda_graph(group, shapes)

        self._cuda_graph_current_step += 1
        return loss

    def _run_param_updates(self, group, shapes, **kwargs):
        if not is_group_tensor_parallel(group):
            for p, s in zip(group["params"], shapes):
                if p.grad is None:
                    continue
                tensor_to_update_p = self._single_param_step(p, s, group, **kwargs)
                self._single_param_update(p, tensor_to_update_p, group)

    def _upload_group_meta_to_cuda_graph(self, group, shapes):  # adjust _group_meta if more meta info changes
        if "lr_tensor" not in self._group_meta:
            self._group_meta["lr_tensor"] = torch.tensor(
                group["lr"],
                dtype=torch.float32,
                device=torch.cuda.current_device()
            )
        else:
            self._group_meta["lr_tensor"].fill_(group["lr"])
        group["lr"] = self._group_meta["lr_tensor"]

    def _offload_group_meta_from_cuda_graph(self, group, shapes):
        group["lr"] = self._group_meta["lr_tensor"].item()
