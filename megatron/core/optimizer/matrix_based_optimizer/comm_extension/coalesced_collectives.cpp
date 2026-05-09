#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

using namespace c10d;

// Simulate allgather(list) using broadcast loop.
// Equivalent to ProcessGroupNCCL::allgather's uneven-size path,
// but WITHOUT calling startCoalescing/endCoalescing internally.
// Each broadcast goes through collective(), which is safe when
// coalescing_state_ is already active (set by outer _coalescing_manager).
//
// outputs: list of N tensors, one per rank, sized to hold gathered data
//          outputs[rank] must already contain the current rank's local data
void allgather_coalesced(
    std::vector<at::Tensor> outputs,
    at::Tensor input,
    c10::intrusive_ptr<ProcessGroup> pg,
    bool async_op) {

    TORCH_CHECK(pg, "ProcessGroup is null");

    int world_size = pg->getSize();
    int rank = pg->getRank();

    TORCH_CHECK(
        static_cast<int>(outputs.size()) == world_size,
        "outputs size must match world_size, got ", outputs.size(),
        " vs world_size ", world_size);

    for (int i = 0; i < world_size; ++i) {
        BroadcastOptions opts;
        opts.rootRank = i;
        opts.rootTensor = 0;
        opts.asyncOp = async_op;
        auto& broadcast_tensor = (i == rank) ? input : outputs[i];
        std::vector<at::Tensor> broadcast_list = {broadcast_tensor};
        pg->broadcast(broadcast_list, opts);
    }
}

// Simulate reduce_scatter(list) using reduce loop.
// Equivalent to ProcessGroupNCCL::reduce_scatter's uneven-size path,
// but WITHOUT calling startCoalescing/endCoalescing internally.
// Note: output and inputs[rank] must share the same underlying memory.
// The caller is responsible for ensuring this (as is the case in
// matrix_based_optimizer where local_data_view is a slice of grad_data
// at the same position as total_data_view[rank]).
void reduce_scatter_coalesced(
    at::Tensor output,
    std::vector<at::Tensor> inputs,
    c10::intrusive_ptr<ProcessGroup> pg,
    int64_t reduce_op,
    bool async_op) {

    TORCH_CHECK(pg, "ProcessGroup is null");

    int world_size = pg->getSize();
    int rank = pg->getRank();

    TORCH_CHECK(
        static_cast<int>(inputs.size()) == world_size,
        "inputs size must match world_size, got ", inputs.size(),
        " vs world_size ", world_size);

    auto nccl_reduce_op = ReduceOp(static_cast<ReduceOp::RedOpType>(reduce_op));

    for (int i = 0; i < world_size; ++i) {
        ReduceOptions opts;
        opts.reduceOp = nccl_reduce_op;
        opts.rootRank = i;
        opts.rootTensor = 0;
        if (i == rank) {
            std::vector<at::Tensor> reduce_list = {output};
            pg->reduce(reduce_list, opts);
        } else {
            std::vector<at::Tensor> reduce_list = {inputs[i]};
            pg->reduce(reduce_list, opts);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "allgather_coalesced",
        &allgather_coalesced,
        "Coalescing-safe allgather using broadcast loop",
        py::arg("outputs"), py::arg("input"), py::arg("pg"), py::arg("async_op"));
    m.def(
        "reduce_scatter_coalesced",
        &reduce_scatter_coalesced,
        "Coalescing-safe reduce_scatter using reduce loop",
        py::arg("output"), py::arg("inputs"), py::arg("pg"),
        py::arg("reduce_op"), py::arg("async_op"));
}
