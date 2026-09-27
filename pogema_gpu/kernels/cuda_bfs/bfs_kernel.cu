#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "bfs_helpers.cuh"

// ─── Unified BFS kernel ────────────────────────────────────────────────────
// FUSED_MODE=true:  computes normalized windows + action codes
// FUSED_MODE=false: computes raw (un-normalized) BFS distances

template <bool FUSED_MODE>
__global__ void bfs_kernel(
    const int8_t*  __restrict__ obstacles,
    const int64_t* __restrict__ agent_pos,
    const int64_t* __restrict__ goal_pos,
    int16_t*       __restrict__ out_windows,
    int32_t*       __restrict__ out_actions,   // unused when !FUSED_MODE
    unsigned*      __restrict__ g_visited,
    int*           __restrict__ g_frontiers,
    int H, int W, int radius, int value_limit, // value_limit unused when !FUSED_MODE
    int bitmap_words, int max_frontier)
{
    const int agent_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;

    const int win_side  = 2 * radius + 1;
    const int win_cells = win_side * win_side;

    extern __shared__ char smem_raw[];
    int*     counters = (int*) smem_raw;
    int16_t* window   = (int16_t*)(counters + 4);

    unsigned* visited    = g_visited + (long long)agent_id * bitmap_words;
    int*      frontier_A = g_frontiers + (long long)agent_id * 2 * max_frontier;
    int*      frontier_B = frontier_A + max_frontier;

    // ── 1. Init ─────────────────────────────────────────────────────────
    if (tid == 0) {
        counters[0] = 0;
        counters[1] = 0;
        counters[2] = 0;
        counters[3] = FUSED_MODE ? -1 : 0;
    }
    __syncthreads();

    int a_row = (int)agent_pos[agent_id * 2];
    int a_col = (int)agent_pos[agent_id * 2 + 1];
    int g_row = (int)goal_pos[agent_id * 2];
    int g_col = (int)goal_pos[agent_id * 2 + 1];

    int win_r_min = a_row - radius;
    int win_c_min = a_col - radius;

    // Obstacles and cells outside the map are final -1 outputs, so count
    // them as resolved.  Otherwise even one obstacle in the requested
    // window prevents the early-exit condition and needlessly exhausts the
    // goal's entire connected component.
    for (int i = tid; i < win_cells; i += nthreads) {
        window[i] = -1;
        int wr = i / win_side;
        int wc = i % win_side;
        int row = win_r_min + wr;
        int col = win_c_min + wc;
        if (row < 0 || row >= H || col < 0 || col >= W || obstacles[row * W + col])
            atomicAdd(&counters[2], 1);
    }
    __syncthreads();

    int goal_idx = g_row * W + g_col;
    if (tid == 0) {
        bit_test_and_set(visited, goal_idx);
        frontier_A[0] = goal_idx;
        counters[0] = 1;

        int wr = g_row - win_r_min;
        int wc = g_col - win_c_min;
        if (wr >= 0 && wr < win_side && wc >= 0 && wc < win_side) {
            window[wr * win_side + wc] = 0;
            atomicAdd(&counters[2], 1);
        }
        if (FUSED_MODE) {
            if (g_row == a_row && g_col == a_col) {
                counters[3] = 0;
            }
        }
    }
    __syncthreads();

    // ── 2. BFS expansion ────────────────────────────────────────────────
    int* cur_front  = frontier_A;
    int* next_front = frontier_B;
    const int total_cells = H * W;

    for (int level = 1; level <= total_cells; ++level) {
        int cur_size = counters[0];
        if (cur_size == 0) break;
        if (counters[2] >= win_cells) break;

        if (tid == 0) counters[1] = 0;
        __syncthreads();

        for (int fi = tid; fi < cur_size; fi += nthreads) {
            int cell = cur_front[fi];
            if (cell < 0 || cell >= total_cells) continue;
            int cr = cell / W;
            int cc = cell % W;

            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int nr, nc;
                if      (d == 0) { nr = cr - 1; nc = cc;     }
                else if (d == 1) { nr = cr + 1; nc = cc;     }
                else if (d == 2) { nr = cr;     nc = cc - 1; }
                else             { nr = cr;     nc = cc + 1; }

                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;

                int nidx = nr * W + nc;
                if (obstacles[nidx]) continue;
                if (!bit_test_and_set(visited, nidx)) continue;

                int slot = atomicAdd(&counters[1], 1);
                if (slot < max_frontier) {
                    next_front[slot] = nidx;
                }

                int wr = nr - win_r_min;
                int wc = nc - win_c_min;
                if (wr >= 0 && wr < win_side && wc >= 0 && wc < win_side) {
                    window[wr * win_side + wc] = (int16_t)level;
                    atomicAdd(&counters[2], 1);
                }

                if (FUSED_MODE) {
                    if (nr == a_row && nc == a_col) {
                        counters[3] = level;
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0 && counters[1] > max_frontier) {
            counters[1] = max_frontier;
        }
        __syncthreads();

        int* tmp = cur_front;
        cur_front  = next_front;
        next_front = tmp;
        if (tid == 0) counters[0] = counters[1];
        __syncthreads();
    }

    if (FUSED_MODE) {
        // ── 3. Next-action code ─────────────────────────────────────────
        int action_code = 0;
        if (tid == 0) {
            int center_dist = counters[3];
            if (center_dist >= 0) {
                int16_t n0 = window[(radius - 1) * win_side + radius];
                int16_t n1 = window[(radius + 1) * win_side + radius];
                int16_t n2 = window[radius * win_side + (radius - 1)];
                int16_t n3 = window[radius * win_side + (radius + 1)];
                if (n0 >= 0 && n0 < center_dist) action_code |= 8;
                if (n1 >= 0 && n1 < center_dist) action_code |= 4;
                if (n2 >= 0 && n2 < center_dist) action_code |= 2;
                if (n3 >= 0 && n3 < center_dist) action_code |= 1;
            }
        }

        // ── 4. Normalize window ─────────────────────────────────────────
        __syncthreads();
        int center_dist = counters[3];

        for (int i = tid; i < win_cells; i += nthreads) {
            int16_t val = window[i];
            if (val == -1) {
                window[i] = (int16_t)(-value_limit * 4);
            } else {
                int norm = (int)val - center_dist;
                if (norm > value_limit) {
                    norm = value_limit * 2;
                } else if (norm < -value_limit) {
                    norm = -value_limit * 2;
                }
                window[i] = (int16_t)norm;
            }
        }
        __syncthreads();

        // ── 5. Write to global memory ───────────────────────────────────
        int16_t* out_win = out_windows + agent_id * win_cells;
        for (int i = tid; i < win_cells; i += nthreads)
            out_win[i] = window[i];

        if (tid == 0)
            out_actions[agent_id] = action_code;
    } else {
        // Raw mode: write distances directly
        int16_t* out_win = out_windows + agent_id * win_cells;
        for (int i = tid; i < win_cells; i += nthreads)
            out_win[i] = window[i];
    }
}

// ─── Shared helpers ─────────────────────────────────────────────────────────

static size_t calc_bfs_smem(int radius) {
    int win_side = 2 * radius + 1;
    return 4 * sizeof(int) + win_side * win_side * sizeof(int16_t);
}

struct BfsBuffers {
    torch::Tensor g_visited;
    torch::Tensor g_frontiers;
    int bitmap_words;
    int max_frontier;
    size_t smem_bytes;
};

static BfsBuffers prepare_bfs_buffers(
    torch::Tensor obstacles, int N, int H, int W, int radius)
{
    int bitmap_words = (H * W + 31) / 32;
    int max_frontier = 2 * (H + W);
    if (max_frontier < 4096) max_frontier = 4096;

    auto g_visited = torch::zeros({(long long)N * bitmap_words},
        torch::TensorOptions().dtype(torch::kInt32).device(obstacles.device()));
    auto g_frontiers = torch::empty({(long long)N * 2 * max_frontier},
        torch::TensorOptions().dtype(torch::kInt32).device(obstacles.device()));

    size_t smem_bytes = calc_bfs_smem(radius);
    return {g_visited, g_frontiers, bitmap_words, max_frontier, smem_bytes};
}

template <bool FUSED_MODE>
static void configure_bfs_dynamic_smem(size_t smem_bytes)
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    TORCH_CHECK(
        err == cudaSuccess,
        "cudaGetDevice failed: ", cudaGetErrorString(err));

    int max_optin = 0;
    err = cudaDeviceGetAttribute(
        &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    TORCH_CHECK(
        err == cudaSuccess,
        "querying opt-in shared-memory limit failed: ", cudaGetErrorString(err));
    TORCH_CHECK(
        smem_bytes <= static_cast<size_t>(max_optin),
        "BFS radius requires ", smem_bytes,
        " bytes of dynamic shared memory, but this GPU supports at most ",
        max_optin, " bytes per block");

    err = cudaFuncSetAttribute(
        bfs_kernel<FUSED_MODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    TORCH_CHECK(
        err == cudaSuccess,
        "opting BFS kernel into large dynamic shared memory failed: ",
        cudaGetErrorString(err));
}

// ─── Launchers ──────────────────────────────────────────────────────────────

torch::Tensor raw_bfs_cost2go(
    torch::Tensor obstacles,
    torch::Tensor agent_pos,
    torch::Tensor goal_pos,
    int H, int W, int radius)
{
    TORCH_CHECK(obstacles.scalar_type() == torch::kInt8,  "obstacles must be int8");
    TORCH_CHECK(agent_pos.scalar_type() == torch::kInt64, "agent_pos must be int64");
    TORCH_CHECK(goal_pos.scalar_type()  == torch::kInt64, "goal_pos must be int64");
    TORCH_CHECK(obstacles.is_contiguous(), "obstacles must be contiguous");
    TORCH_CHECK(agent_pos.is_contiguous(), "agent_pos must be contiguous");
    TORCH_CHECK(goal_pos.is_contiguous(),  "goal_pos must be contiguous");

    int N = agent_pos.size(0);
    int win_side = 2 * radius + 1;
    auto out_windows = torch::empty({N, win_side * win_side},
        torch::TensorOptions().dtype(torch::kInt16).device(agent_pos.device()));

    if (N == 0) return out_windows;

    auto bufs = prepare_bfs_buffers(obstacles, N, H, W, radius);
    configure_bfs_dynamic_smem<false>(bufs.smem_bytes);

    bfs_kernel<false><<<N, 128, bufs.smem_bytes>>>(
        obstacles.data_ptr<int8_t>(),
        agent_pos.data_ptr<int64_t>(),
        goal_pos.data_ptr<int64_t>(),
        out_windows.data_ptr<int16_t>(),
        nullptr,  // out_actions unused
        (unsigned*)bufs.g_visited.data_ptr<int32_t>(),
        bufs.g_frontiers.data_ptr<int32_t>(),
        H, W, radius, 0,  // value_limit unused
        bufs.bitmap_words, bufs.max_frontier);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("raw_bfs launch: ") + cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("raw_bfs exec: ") + cudaGetErrorString(err));

    return out_windows;
}

std::vector<torch::Tensor> fused_bfs_cost2go(
    torch::Tensor obstacles,
    torch::Tensor agent_pos,
    torch::Tensor goal_pos,
    int H, int W, int radius, int value_limit)
{
    TORCH_CHECK(obstacles.scalar_type() == torch::kInt8,  "obstacles must be int8");
    TORCH_CHECK(agent_pos.scalar_type() == torch::kInt64, "agent_pos must be int64");
    TORCH_CHECK(goal_pos.scalar_type()  == torch::kInt64, "goal_pos must be int64");
    TORCH_CHECK(obstacles.is_contiguous(), "obstacles must be contiguous");
    TORCH_CHECK(agent_pos.is_contiguous(), "agent_pos must be contiguous");
    TORCH_CHECK(goal_pos.is_contiguous(),  "goal_pos must be contiguous");

    int N = agent_pos.size(0);
    int win_side = 2 * radius + 1;
    auto out_windows = torch::empty({N, win_side * win_side},
        torch::TensorOptions().dtype(torch::kInt16).device(agent_pos.device()));
    auto out_actions = torch::empty({N},
        torch::TensorOptions().dtype(torch::kInt32).device(agent_pos.device()));

    if (N == 0) return {out_windows, out_actions};

    auto bufs = prepare_bfs_buffers(obstacles, N, H, W, radius);
    configure_bfs_dynamic_smem<true>(bufs.smem_bytes);

    bfs_kernel<true><<<N, 128, bufs.smem_bytes>>>(
        obstacles.data_ptr<int8_t>(),
        agent_pos.data_ptr<int64_t>(),
        goal_pos.data_ptr<int64_t>(),
        out_windows.data_ptr<int16_t>(),
        out_actions.data_ptr<int32_t>(),
        (unsigned*)bufs.g_visited.data_ptr<int32_t>(),
        bufs.g_frontiers.data_ptr<int32_t>(),
        H, W, radius, value_limit,
        bufs.bitmap_words, bufs.max_frontier);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("bfs launch: ") + cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("bfs exec: ") + cudaGetErrorString(err));

    return {out_windows, out_actions};
}
