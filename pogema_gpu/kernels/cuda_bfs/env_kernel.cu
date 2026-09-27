#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void env_step_part1_kernel(
    const int64_t* __restrict__ pos, const int64_t* __restrict__ actions,
    const int32_t* __restrict__ moves, const bool* __restrict__ grid,
    int32_t* __restrict__ next_flat, int32_t* __restrict__ who_was_at,
    int N, int H, int W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int r = pos[i * 2];
    int c = pos[i * 2 + 1];
    int action = actions[i];

    int nr = r + moves[action * 2];
    int nc = c + moves[action * 2 + 1];

    bool hit = (nr < 0 || nr >= H || nc < 0 || nc >= W || grid[nr * W + nc]);

    int flat = r * W + c;
    int nflat = hit ? flat : (nr * W + nc);

    next_flat[i] = nflat;
    who_was_at[flat] = i; 
}

__global__ void env_step_swaps_kernel(
    const int64_t* __restrict__ pos, int32_t* __restrict__ next_flat,
    const int32_t* __restrict__ who_was_at, int N, int W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int flat = pos[i * 2] * W + pos[i * 2 + 1];
    int nflat = next_flat[i];

    if (nflat != flat) {
        int target_occupant = who_was_at[nflat];
        if (target_occupant != -1 && target_occupant != i) {
            if (next_flat[target_occupant] == flat) {
                next_flat[i] = flat; 
            }
        }
    }
}

__global__ void cascade_write_claimants_kernel(
    const int64_t* __restrict__ pos, const int32_t* __restrict__ next_flat,
    int32_t* __restrict__ claimants, int N, int W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int flat = pos[i * 2] * W + pos[i * 2 + 1];
    int nflat = next_flat[i];
    bool is_moving = (nflat != flat);

    int priority = is_moving ? (i + N) : i;
    atomicMin(&claimants[nflat], priority);
}

__global__ void cascade_resolve_kernel(
    const int64_t* __restrict__ pos, int32_t* __restrict__ next_flat,
    const int32_t* __restrict__ claimants, int32_t* __restrict__ d_changed,
    int N, int W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int flat = pos[i * 2] * W + pos[i * 2 + 1];
    int nflat = next_flat[i];
    bool is_moving = (nflat != flat);

    if (is_moving) {
        int priority = i + N;
        if (claimants[nflat] != priority) {
            next_flat[i] = flat;
            *d_changed = 1; 
        }
    }
}

__global__ void env_finalize_kernel(
    int64_t* __restrict__ pos, const int32_t* __restrict__ next_flat,
    const int64_t* __restrict__ goals, int32_t* __restrict__ d_all_on_goal,
    int32_t* __restrict__ solve_time, int current_step,
    int N, int W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int nflat = next_flat[i];
    int nr = nflat / W;
    int nc = nflat % W;

    pos[i * 2] = nr;
    pos[i * 2 + 1] = nc;

    if (nr == goals[i * 2] && nc == goals[i * 2 + 1]) {
        if (solve_time[i] == -1) {
            solve_time[i] = current_step;
        }
    } else {
        solve_time[i] = -1;
        *d_all_on_goal = 0;
    }
}

bool env_step_cuda(
    torch::Tensor pos, torch::Tensor actions, torch::Tensor grid, torch::Tensor goals,
    torch::Tensor moves, torch::Tensor next_flat, torch::Tensor who_was_at,
    torch::Tensor claimants, torch::Tensor d_changed, torch::Tensor d_all_on_goal,
    torch::Tensor solve_time, int current_step
) {
    int N = pos.size(0);
    int H = grid.size(0);
    int W = grid.size(1);
    int blocks = (N + 127) / 128;
    int threads = 128;

    cudaMemset(who_was_at.data_ptr<int32_t>(), 0xFF, H * W * sizeof(int32_t));

    env_step_part1_kernel<<<blocks, threads>>>(
        pos.data_ptr<int64_t>(), actions.data_ptr<int64_t>(),
        moves.data_ptr<int32_t>(), grid.data_ptr<bool>(),
        next_flat.data_ptr<int32_t>(), who_was_at.data_ptr<int32_t>(), N, H, W
    );

    env_step_swaps_kernel<<<blocks, threads>>>(
        pos.data_ptr<int64_t>(), next_flat.data_ptr<int32_t>(),
        who_was_at.data_ptr<int32_t>(), N, W
    );

    int h_changed = 1;
    while (h_changed) {
        h_changed = 0;
        cudaMemcpy(d_changed.data_ptr<int32_t>(), &h_changed, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemset(claimants.data_ptr<int32_t>(), 0x7F, H * W * sizeof(int32_t)); 

        cascade_write_claimants_kernel<<<blocks, threads>>>(
            pos.data_ptr<int64_t>(), next_flat.data_ptr<int32_t>(), claimants.data_ptr<int32_t>(), N, W
        );

        cascade_resolve_kernel<<<blocks, threads>>>(
            pos.data_ptr<int64_t>(), next_flat.data_ptr<int32_t>(), claimants.data_ptr<int32_t>(), d_changed.data_ptr<int32_t>(), N, W
        );

        cudaMemcpy(&h_changed, d_changed.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    int h_all_on_goal = 1;
    cudaMemcpy(d_all_on_goal.data_ptr<int32_t>(), &h_all_on_goal, sizeof(int32_t), cudaMemcpyHostToDevice);

    env_finalize_kernel<<<blocks, threads>>>(
        pos.data_ptr<int64_t>(), next_flat.data_ptr<int32_t>(), goals.data_ptr<int64_t>(), 
        d_all_on_goal.data_ptr<int32_t>(), solve_time.data_ptr<int32_t>(), current_step, 
        N, W
    );

    cudaMemcpy(&h_all_on_goal, d_all_on_goal.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    return h_all_on_goal == 1;
}