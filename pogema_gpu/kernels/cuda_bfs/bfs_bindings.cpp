#include <torch/extension.h>
#include <cstdint>

std::vector<torch::Tensor> fused_bfs_cost2go(
    torch::Tensor obstacles, torch::Tensor agent_pos, torch::Tensor goal_pos,
    int H, int W, int radius, int value_limit);

torch::Tensor raw_bfs_cost2go(
    torch::Tensor obstacles, torch::Tensor agent_pos, torch::Tensor goal_pos,
    int H, int W, int radius);

std::vector<torch::Tensor> extract_and_normalize_cuda(
    torch::Tensor cache_windows, torch::Tensor pos, torch::Tensor cache_center,
    int value_limit, int obs_radius, int cache_radius);

torch::Tensor get_neighbors_cuda(
    torch::Tensor pos, torch::Tensor goals, torch::Tensor history,
    torch::Tensor next_actions, torch::Tensor coord_lookup,
    int agents_radius, int limit, int coord_offset,
    int pad_token, int inf_dist, int num_hist, int num_neighbors);

torch::Tensor get_neighbors_spatial_cuda(
    torch::Tensor pos, torch::Tensor goals, torch::Tensor history,
    torch::Tensor next_actions, torch::Tensor coord_lookup,
    int H, int W, int agents_radius, int limit, int coord_offset,
    int pad_token, int num_hist, int num_neighbors);

torch::Tensor get_neighbors_spatial_sharded_cuda(
    torch::Tensor ego_pos, torch::Tensor pos, torch::Tensor goals,
    torch::Tensor history, torch::Tensor next_actions, torch::Tensor coord_lookup,
    int H, int W, int agents_radius, int limit, int coord_offset,
    int pad_token, int num_hist, int num_neighbors);

torch::Tensor get_chat_neighbors_spatial_cuda(
    torch::Tensor pos, int H, int W, int agents_radius, int max_neighbors);

torch::Tensor get_chat_neighbors_spatial_sharded_cuda(
    torch::Tensor ego_pos, torch::Tensor pos, int H, int W,
    int agents_radius, int max_neighbors);

bool env_step_cuda(
    torch::Tensor pos, torch::Tensor actions, torch::Tensor grid, torch::Tensor goals,
    torch::Tensor moves, torch::Tensor next_flat, torch::Tensor who_was_at,
    torch::Tensor claimants, torch::Tensor d_changed, torch::Tensor d_all_on_goal, 
    torch::Tensor solve_time, int current_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bfs_cost2go", &fused_bfs_cost2go, "Fused BFS cost-to-go with normalization and action codes");
    m.def("raw_bfs_cost2go", &raw_bfs_cost2go, "Raw BFS distances without normalization");
    m.def("extract_and_normalize_cuda", &extract_and_normalize_cuda, "Extracts and normalizes windows from cache");
    m.def("get_neighbors_cuda", &get_neighbors_cuda, "Gather tokens for nearest neighbors");
    m.def("get_neighbors_spatial_cuda", &get_neighbors_spatial_cuda, "Gather nearest-neighbor tokens with a spatial cell list");
    m.def("get_neighbors_spatial_sharded_cuda", &get_neighbors_spatial_sharded_cuda, "Gather local receiver tokens from global spatial cell lists");
    m.def("get_chat_neighbors_spatial_cuda", &get_chat_neighbors_spatial_cuda, "Gather communication neighbor ids with a spatial cell list");
    m.def("get_chat_neighbors_spatial_sharded_cuda", &get_chat_neighbors_spatial_sharded_cuda, "Gather local receiver communication ids from global positions");
    m.def("env_step_cuda", &env_step_cuda, "Execute one step of MiniPogema entirely on GPU");
}
