#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void extract_and_normalize_kernel(
    const int16_t* __restrict__ cache_windows,
    const int64_t* __restrict__ pos,
    const int64_t* __restrict__ cache_center,
    int16_t* __restrict__ out_windows,
    int32_t* __restrict__ out_actions,
    int value_limit, int obs_radius, int cache_radius
) {
    int agent_id = blockIdx.x;
    int tid = threadIdx.x;

    int obs_side = 2 * obs_radius + 1;
    int obs_cells = obs_side * obs_side;
    int cache_side = 2 * cache_radius + 1;
    long long cache_cells = (long long)cache_side * cache_side;

    int a_row = (int)pos[agent_id * 2];
    int a_col = (int)pos[agent_id * 2 + 1];
    int c_row = (int)cache_center[agent_id * 2];
    int c_col = (int)cache_center[agent_id * 2 + 1];

    int offset_r = a_row - c_row;
    int offset_c = a_col - c_col;

    int center_cr = cache_radius + offset_r;
    int center_cc = cache_radius + offset_c;

    const int16_t* agent_cache = cache_windows + (long long)agent_id * cache_cells;
    int center_dist = agent_cache[center_cr * cache_side + center_cc];

    if (tid == 0) {
        int action_code = 0;
        if (center_dist >= 0) {
            int16_t n_up    = agent_cache[(center_cr - 1) * cache_side + center_cc];
            int16_t n_down  = agent_cache[(center_cr + 1) * cache_side + center_cc];
            int16_t n_left  = agent_cache[center_cr * cache_side + (center_cc - 1)];
            int16_t n_right = agent_cache[center_cr * cache_side + (center_cc + 1)];

            if (n_up >= 0 && n_up < center_dist)       action_code |= 8;
            if (n_down >= 0 && n_down < center_dist)   action_code |= 4;
            if (n_left >= 0 && n_left < center_dist)   action_code |= 2;
            if (n_right >= 0 && n_right < center_dist) action_code |= 1;
        }
        out_actions[agent_id] = action_code;
    }

    int16_t* agent_out = out_windows + agent_id * obs_cells;
    for (int i = tid; i < obs_cells; i += blockDim.x) {
        int wr = i / obs_side;
        int wc = i % obs_side;
        int cr = center_cr - obs_radius + wr;
        int cc = center_cc - obs_radius + wc;

        int16_t raw_val = agent_cache[cr * cache_side + cc];

        if (raw_val == -1) {
            agent_out[i] = (int16_t)(-value_limit * 4);
        } else {
            int norm = (int)raw_val - center_dist;
            if (norm > value_limit) norm = value_limit * 2;
            else if (norm < -value_limit) norm = -value_limit * 2;
            agent_out[i] = (int16_t)norm;
        }
    }
}

std::vector<torch::Tensor> extract_and_normalize_cuda(
    torch::Tensor cache_windows,
    torch::Tensor pos,
    torch::Tensor cache_center,
    int value_limit, int obs_radius, int cache_radius)
{
    TORCH_CHECK(cache_windows.scalar_type() == torch::kInt16, "cache_windows must be int16");
    TORCH_CHECK(pos.scalar_type() == torch::kInt64, "pos must be int64");
    TORCH_CHECK(cache_center.scalar_type() == torch::kInt64, "cache_center must be int64");

    int N = pos.size(0);
    int obs_side = 2 * obs_radius + 1;

    auto out_windows = torch::empty({N, obs_side * obs_side},
        torch::TensorOptions().dtype(torch::kInt16).device(pos.device()));
    auto out_actions = torch::empty({N},
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));

    if (N > 0) {
        extract_and_normalize_kernel<<<N, 128>>>(
            cache_windows.data_ptr<int16_t>(),
            pos.data_ptr<int64_t>(),
            cache_center.data_ptr<int64_t>(),
            out_windows.data_ptr<int16_t>(),
            out_actions.data_ptr<int32_t>(),
            value_limit, obs_radius, cache_radius
        );
    }

    return {out_windows, out_actions};
}
