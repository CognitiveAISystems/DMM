#include <torch/extension.h>

std::vector<torch::Tensor> cuda_pibt_resolve(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    int64_t num_cells);

std::vector<torch::Tensor> cuda_pibt_resolve_batched(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells);

std::vector<torch::Tensor> cuda_pibt_resolve_batched_warp(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells);

std::vector<torch::Tensor> cuda_pibt_resolve_batched_parallel_occupancy(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells);

std::vector<torch::Tensor> cuda_pibt_resolve_batched_compact(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells);

std::vector<torch::Tensor> cuda_pibt_resolve_batched_hybrid(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells);

std::vector<torch::Tensor> cuda_pibt_resolve_batched_components(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("resolve", &cuda_pibt_resolve, "Exact DMM PIBT resolve (CUDA)");
  module.def(
      "resolve_batched",
      &cuda_pibt_resolve_batched,
      "Batched exact DMM PIBT resolve (CUDA)");
  module.def(
      "resolve_batched_warp",
      &cuda_pibt_resolve_batched_warp,
      "Warp-assisted batched exact DMM PIBT resolve (CUDA)");
  module.def(
      "resolve_batched_parallel_occupancy",
      &cuda_pibt_resolve_batched_parallel_occupancy,
      "Batched exact DMM PIBT with parallel occupancy build (CUDA)");
  module.def(
      "resolve_batched_compact",
      &cuda_pibt_resolve_batched_compact,
      "Priority-renamed compact exact DMM PIBT resolve (CUDA)");
  module.def(
      "resolve_batched_hybrid",
      &cuda_pibt_resolve_batched_hybrid,
      "GPU-packed CPU-executed exact DMM PIBT resolve");
  module.def(
      "resolve_batched_components",
      &cuda_pibt_resolve_batched_components,
      "Component-parallel exact DMM PIBT resolve (CUDA)");
}
