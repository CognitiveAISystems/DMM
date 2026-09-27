#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define MAX_NEIGHBORS 13

__global__ void build_cell_lists_kernel(
    const int64_t* __restrict__ pos,
    int32_t* __restrict__ cell_heads,
    int32_t* __restrict__ next_agent,
    int N,
    int cell_size,
    int num_cell_rows,
    int num_cell_cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int r = (int)pos[i * 2];
    int c = (int)pos[i * 2 + 1];
    if (r < 0 || c < 0) {
        next_agent[i] = -1;
        return;
    }
    int cell_r = r / cell_size;
    int cell_c = c / cell_size;
    if (cell_r < 0 || cell_r >= num_cell_rows ||
        cell_c < 0 || cell_c >= num_cell_cols) {
        next_agent[i] = -1;
        return;
    }
    int cell_id = cell_r * num_cell_cols + cell_c;
    next_agent[i] = atomicExch(&cell_heads[cell_id], i);
}

__device__ __forceinline__ void insert_top_neighbor(
    long long score,
    int agent_id,
    long long* top_scores,
    int* top_ids,
    int k_neighbors
) {
    for (int k = 0; k < k_neighbors; ++k) {
        if (score < top_scores[k]) {
            for (int m = k_neighbors - 1; m > k; --m) {
                top_scores[m] = top_scores[m - 1];
                top_ids[m] = top_ids[m - 1];
            }
            top_scores[k] = score;
            top_ids[k] = agent_id;
            break;
        }
    }
}

__global__ void get_neighbors_kernel(
    const int64_t* __restrict__ pos,
    const int64_t* __restrict__ goals,
    const int64_t* __restrict__ history,
    const int32_t* __restrict__ next_actions,
    const int64_t* __restrict__ coord_lookup,
    int64_t* __restrict__ out_obs,
    int N, 
    int agents_radius, 
    int limit, 
    int coord_offset,
    int pad_token, 
    int inf_dist, 
    int num_hist, 
    int num_neighbors
) {
    int ego_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ego_id >= N) return;

    int k_neighbors = min(num_neighbors, MAX_NEIGHBORS);

    int ego_r = pos[ego_id * 2];
    int ego_c = pos[ego_id * 2 + 1];

    long long top_scores[MAX_NEIGHBORS];
    int top_ids[MAX_NEIGHBORS];
    
    for (int i = 0; i < k_neighbors; ++i) {
        top_scores[i] = 999999999999LL;
        top_ids[i] = -1;
    }

    for (int j = 0; j < N; ++j) {
        int dr = pos[j * 2] - ego_r;
        int dc = pos[j * 2 + 1] - ego_c;
        int abs_dr = abs(dr);
        int abs_dc = abs(dc);
        int cheb = max(abs_dr, abs_dc);

        if (cheb <= agents_radius) {
            int manh = abs_dr + abs_dc;
            long long score = (long long)manh * N + j; 

            insert_top_neighbor(score, j, top_scores, top_ids, k_neighbors);
        }
    }

    int tokens_per_neighbor = 5 + num_hist; 
    int64_t* out_ptr = out_obs + ego_id * (num_neighbors * tokens_per_neighbor);

    for (int i = 0; i < k_neighbors; ++i) {
        int n_id = top_ids[i];
        int base_idx = i * tokens_per_neighbor;

        if (n_id != -1) {
            // Rel Pos
            int rr = max(-limit, min(limit, (int)pos[n_id * 2] - ego_r));
            int rc = max(-limit, min(limit, (int)pos[n_id * 2 + 1] - ego_c));
            out_ptr[base_idx + 0] = coord_lookup[rr + coord_offset];
            out_ptr[base_idx + 1] = coord_lookup[rc + coord_offset];

            // Rel Goal
            int gr = max(-limit, min(limit, (int)goals[n_id * 2] - ego_r));
            int gc = max(-limit, min(limit, (int)goals[n_id * 2 + 1] - ego_c));
            out_ptr[base_idx + 2] = coord_lookup[gr + coord_offset];
            out_ptr[base_idx + 3] = coord_lookup[gc + coord_offset];

            // History
            for (int h = 0; h < num_hist; ++h) {
                out_ptr[base_idx + 4 + h] = history[n_id * num_hist + h];
            }

            // Next Action
            out_ptr[base_idx + 4 + num_hist] = next_actions[n_id];
        } else {
            // Padding
            for (int p = 0; p < tokens_per_neighbor; ++p) {
                out_ptr[base_idx + p] = pad_token;
            }
        }
    }
    for (int i = k_neighbors; i < num_neighbors; ++i) {
        int base_idx = i * tokens_per_neighbor;
        for (int p = 0; p < tokens_per_neighbor; ++p) {
            out_ptr[base_idx + p] = pad_token;
        }
    }
}

__global__ void get_neighbors_spatial_kernel(
    const int64_t* __restrict__ ego_pos,
    const int64_t* __restrict__ pos,
    const int64_t* __restrict__ goals,
    const int64_t* __restrict__ history,
    const int32_t* __restrict__ next_actions,
    const int64_t* __restrict__ coord_lookup,
    const int32_t* __restrict__ cell_heads,
    const int32_t* __restrict__ next_agent,
    int64_t* __restrict__ out_obs,
    int ego_N, int N, int H, int W, int agents_radius, int limit, int coord_offset,
    int pad_token, int num_hist, int num_neighbors, int cell_size,
    int num_cell_rows, int num_cell_cols
) {
    int ego_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ego_id >= ego_N) return;
    int k_neighbors = min(num_neighbors, MAX_NEIGHBORS);
    int ego_r = (int)ego_pos[ego_id * 2];
    int ego_c = (int)ego_pos[ego_id * 2 + 1];
    long long top_scores[MAX_NEIGHBORS];
    int top_ids[MAX_NEIGHBORS];
    for (int i = 0; i < k_neighbors; ++i) {
        top_scores[i] = 999999999999LL;
        top_ids[i] = -1;
    }
    int ego_cell_r = ego_r / cell_size;
    int ego_cell_c = ego_c / cell_size;
    int r0 = max(0, ego_cell_r - 1);
    int r1 = min(num_cell_rows - 1, ego_cell_r + 1);
    int c0 = max(0, ego_cell_c - 1);
    int c1 = min(num_cell_cols - 1, ego_cell_c + 1);
    for (int cell_r = r0; cell_r <= r1; ++cell_r) {
        for (int cell_c = c0; cell_c <= c1; ++cell_c) {
            int j = cell_heads[cell_r * num_cell_cols + cell_c];
            while (j != -1) {
                int dr = (int)pos[j * 2] - ego_r;
                int dc = (int)pos[j * 2 + 1] - ego_c;
                if (max(abs(dr), abs(dc)) <= agents_radius) {
                    long long score = (long long)(abs(dr) + abs(dc)) * N + j;
                    insert_top_neighbor(score, j, top_scores, top_ids, k_neighbors);
                }
                j = next_agent[j];
            }
        }
    }
    int tokens_per_neighbor = 5 + num_hist;
    int64_t* out_ptr = out_obs + ego_id * (num_neighbors * tokens_per_neighbor);
    for (int i = 0; i < num_neighbors; ++i) {
        int n_id = i < k_neighbors ? top_ids[i] : -1;
        int base = i * tokens_per_neighbor;
        if (n_id < 0) {
            for (int p = 0; p < tokens_per_neighbor; ++p) out_ptr[base + p] = pad_token;
            continue;
        }
        int rr = max(-limit, min(limit, (int)pos[n_id * 2] - ego_r));
        int rc = max(-limit, min(limit, (int)pos[n_id * 2 + 1] - ego_c));
        int gr = max(-limit, min(limit, (int)goals[n_id * 2] - ego_r));
        int gc = max(-limit, min(limit, (int)goals[n_id * 2 + 1] - ego_c));
        out_ptr[base] = coord_lookup[rr + coord_offset];
        out_ptr[base + 1] = coord_lookup[rc + coord_offset];
        out_ptr[base + 2] = coord_lookup[gr + coord_offset];
        out_ptr[base + 3] = coord_lookup[gc + coord_offset];
        for (int h = 0; h < num_hist; ++h)
            out_ptr[base + 4 + h] = history[n_id * num_hist + h];
        out_ptr[base + 4 + num_hist] = next_actions[n_id];
    }
}

__global__ void get_chat_neighbors_spatial_kernel(
    const int64_t* __restrict__ ego_pos,
    const int64_t* __restrict__ pos,
    const int32_t* __restrict__ cell_heads,
    const int32_t* __restrict__ next_agent,
    int64_t* __restrict__ output,
    int ego_N, int N, int agents_radius, int max_neighbors, int cell_size,
    int num_cell_rows, int num_cell_cols
) {
    int ego_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ego_id >= ego_N) return;
    int k_neighbors = min(max_neighbors, MAX_NEIGHBORS);
    int ego_r = (int)ego_pos[ego_id * 2];
    int ego_c = (int)ego_pos[ego_id * 2 + 1];
    long long top_scores[MAX_NEIGHBORS];
    int top_ids[MAX_NEIGHBORS];
    for (int i = 0; i < k_neighbors; ++i) {
        top_scores[i] = 999999999999LL;
        top_ids[i] = -1;
    }
    int ego_cell_r = ego_r / cell_size;
    int ego_cell_c = ego_c / cell_size;
    for (int cr = max(0, ego_cell_r - 1); cr <= min(num_cell_rows - 1, ego_cell_r + 1); ++cr) {
        for (int cc = max(0, ego_cell_c - 1); cc <= min(num_cell_cols - 1, ego_cell_c + 1); ++cc) {
            int j = cell_heads[cr * num_cell_cols + cc];
            while (j != -1) {
                int dr = (int)pos[j * 2] - ego_r;
                int dc = (int)pos[j * 2 + 1] - ego_c;
                if (max(abs(dr), abs(dc)) <= agents_radius) {
                    long long score = (long long)(abs(dr) + abs(dc)) * N + j;
                    insert_top_neighbor(score, j, top_scores, top_ids, k_neighbors);
                }
                j = next_agent[j];
            }
        }
    }
    int64_t* out = output + ego_id * max_neighbors;
    for (int i = 0; i < max_neighbors; ++i)
        out[i] = i < k_neighbors ? top_ids[i] : -1;
}

static void build_cell_lists(
    torch::Tensor pos, torch::Tensor heads, torch::Tensor next_agent,
    int N, int cell_size, int rows, int cols
) {
    int threads = 128;
    build_cell_lists_kernel<<<(N + threads - 1) / threads, threads>>>(
        pos.data_ptr<int64_t>(), heads.data_ptr<int32_t>(),
        next_agent.data_ptr<int32_t>(), N, cell_size, rows, cols);
}

torch::Tensor get_neighbors_cuda(
    torch::Tensor pos, torch::Tensor goals, torch::Tensor history,
    torch::Tensor next_actions, torch::Tensor coord_lookup,
    int agents_radius, int limit, int coord_offset,
    int pad_token, int inf_dist, int num_hist, int num_neighbors
) {
    int N = pos.size(0);
    int tokens_per_neighbor = 5 + num_hist;
    
    auto out_obs = torch::empty({N, num_neighbors * tokens_per_neighbor},
        torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));

    if (N > 0) {
        int threads = 128;
        int blocks = (N + threads - 1) / threads;
        get_neighbors_kernel<<<blocks, threads>>>(
            pos.data_ptr<int64_t>(),
            goals.data_ptr<int64_t>(),
            history.data_ptr<int64_t>(),
            next_actions.data_ptr<int32_t>(),
            coord_lookup.data_ptr<int64_t>(),
            out_obs.data_ptr<int64_t>(),
            N, agents_radius, limit, coord_offset, 
            pad_token, inf_dist, num_hist, num_neighbors
        );
    }
    return out_obs;
}

torch::Tensor get_neighbors_spatial_cuda(
    torch::Tensor pos, torch::Tensor goals, torch::Tensor history,
    torch::Tensor next_actions, torch::Tensor coord_lookup,
    int H, int W, int agents_radius, int limit, int coord_offset,
    int pad_token, int num_hist, int num_neighbors
) {
    int N = pos.size(0);
    auto output = torch::empty({N, num_neighbors * (5 + num_hist)},
        torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));
    if (N == 0) return output;
    int cell_size = agents_radius + 1 > 1 ? agents_radius + 1 : 1;
    int rows = (H + cell_size - 1) / cell_size;
    int cols = (W + cell_size - 1) / cell_size;
    auto heads = torch::full({rows * cols}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    auto next_agent = torch::full({N}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    build_cell_lists(pos, heads, next_agent, N, cell_size, rows, cols);
    int threads = 128;
    get_neighbors_spatial_kernel<<<(N + threads - 1) / threads, threads>>>(
        pos.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), goals.data_ptr<int64_t>(),
        history.data_ptr<int64_t>(), next_actions.data_ptr<int32_t>(),
        coord_lookup.data_ptr<int64_t>(), heads.data_ptr<int32_t>(),
        next_agent.data_ptr<int32_t>(), output.data_ptr<int64_t>(),
        N, N, H, W, agents_radius, limit, coord_offset, pad_token, num_hist,
        num_neighbors, cell_size, rows, cols);
    return output;
}

torch::Tensor get_neighbors_spatial_sharded_cuda(
    torch::Tensor ego_pos, torch::Tensor pos, torch::Tensor goals,
    torch::Tensor history, torch::Tensor next_actions, torch::Tensor coord_lookup,
    int H, int W, int agents_radius, int limit, int coord_offset,
    int pad_token, int num_hist, int num_neighbors
) {
    int ego_N = ego_pos.size(0);
    int N = pos.size(0);
    auto output = torch::empty({ego_N, num_neighbors * (5 + num_hist)},
        torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));
    if (ego_N == 0) return output;
    int cell_size = agents_radius + 1 > 1 ? agents_radius + 1 : 1;
    int rows = (H + cell_size - 1) / cell_size;
    int cols = (W + cell_size - 1) / cell_size;
    auto heads = torch::full({rows * cols}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    auto next_agent = torch::full({N}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    build_cell_lists(pos, heads, next_agent, N, cell_size, rows, cols);
    int threads = 128;
    get_neighbors_spatial_kernel<<<(ego_N + threads - 1) / threads, threads>>>(
        ego_pos.data_ptr<int64_t>(), pos.data_ptr<int64_t>(),
        goals.data_ptr<int64_t>(), history.data_ptr<int64_t>(),
        next_actions.data_ptr<int32_t>(), coord_lookup.data_ptr<int64_t>(),
        heads.data_ptr<int32_t>(), next_agent.data_ptr<int32_t>(),
        output.data_ptr<int64_t>(), ego_N, N, H, W, agents_radius, limit,
        coord_offset, pad_token, num_hist, num_neighbors, cell_size, rows, cols);
    return output;
}

torch::Tensor get_chat_neighbors_spatial_cuda(
    torch::Tensor pos, int H, int W, int agents_radius, int max_neighbors
) {
    int N = pos.size(0);
    auto output = torch::full({N, max_neighbors}, -1,
        torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));
    if (N == 0) return output;
    int cell_size = agents_radius + 1 > 1 ? agents_radius + 1 : 1;
    int rows = (H + cell_size - 1) / cell_size;
    int cols = (W + cell_size - 1) / cell_size;
    auto heads = torch::full({rows * cols}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    auto next_agent = torch::full({N}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    build_cell_lists(pos, heads, next_agent, N, cell_size, rows, cols);
    int threads = 128;
    get_chat_neighbors_spatial_kernel<<<(N + threads - 1) / threads, threads>>>(
        pos.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), heads.data_ptr<int32_t>(),
        next_agent.data_ptr<int32_t>(), output.data_ptr<int64_t>(), N,
        N, agents_radius, max_neighbors, cell_size, rows, cols);
    return output;
}

torch::Tensor get_chat_neighbors_spatial_sharded_cuda(
    torch::Tensor ego_pos, torch::Tensor pos, int H, int W,
    int agents_radius, int max_neighbors
) {
    int ego_N = ego_pos.size(0);
    int N = pos.size(0);
    auto output = torch::full({ego_N, max_neighbors}, -1,
        torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));
    if (ego_N == 0) return output;
    int cell_size = agents_radius + 1 > 1 ? agents_radius + 1 : 1;
    int rows = (H + cell_size - 1) / cell_size;
    int cols = (W + cell_size - 1) / cell_size;
    auto heads = torch::full({rows * cols}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    auto next_agent = torch::full({N}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(pos.device()));
    build_cell_lists(pos, heads, next_agent, N, cell_size, rows, cols);
    int threads = 128;
    get_chat_neighbors_spatial_kernel<<<(ego_N + threads - 1) / threads, threads>>>(
        ego_pos.data_ptr<int64_t>(), pos.data_ptr<int64_t>(),
        heads.data_ptr<int32_t>(), next_agent.data_ptr<int32_t>(),
        output.data_ptr<int64_t>(), ego_N, N, agents_radius, max_neighbors,
        cell_size, rows, cols);
    return output;
}
