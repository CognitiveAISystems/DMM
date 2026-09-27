#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <limits>
namespace {
__device__ void pibt_environment(
    const int64_t* current,
    const int64_t* candidates,
    const int64_t* preferences,
    const bool* valid,
    const int64_t* order,
    const int64_t* num_cells,
    int64_t* occupied_now,
    int64_t* occupied_next,
    int64_t* next_ids,
    int64_t* actions,
    int64_t* stack_agents,
    int64_t* stack_offsets,
    int64_t num_envs,
    int64_t num_agents,
    int64_t max_num_cells,
    bool populate_occupancy, int64_t env) {

  const int64_t agent_base = env * num_agents;
  const int64_t action_base = agent_base * 5;
  const int64_t cell_base = env * max_num_cells;
  const int64_t cells = num_cells[env];

  if (populate_occupancy) {
    for (int64_t agent = 0; agent < num_agents; ++agent) {
      const int64_t cell = current[agent_base + agent];
      if (cell >= 0 && cell < cells) occupied_now[cell_base + cell] = agent;
    }
  }

  for (int64_t root_index = 0; root_index < num_agents; ++root_index) {
    const int64_t root = order[agent_base + root_index];
    if (next_ids[agent_base + root] != -1) continue;
    int64_t stack_top = 0;
    stack_agents[agent_base] = root;
    stack_offsets[agent_base] = 0;

    while (stack_top >= 0) {
      const int64_t stack_index = agent_base + stack_top;
      const int64_t agent = stack_agents[stack_index];
      bool descended = false;
      bool completed_chain = false;

      while (stack_offsets[stack_index] < 5) {
        const int64_t rank = stack_offsets[stack_index]++;
        const int64_t pref_index = action_base + agent * 5 + rank;
        const int64_t action = preferences[pref_index];
        const int64_t av_index = action_base + agent * 5 + action;
        if (action < 0 || action >= 5 || !valid[av_index]) continue;
        const int64_t target = candidates[av_index];
        if (target < 0 || target >= cells ||
            occupied_next[cell_base + target] != -1) continue;
        const int64_t occupant = occupied_now[cell_base + target];
        if (occupant != -1 &&
            next_ids[agent_base + occupant] == current[agent_base + agent]) {
          continue;
        }

        occupied_next[cell_base + target] = agent;
        next_ids[agent_base + agent] = target;
        actions[agent_base + agent] = action;
        if (occupant != -1 && target != current[agent_base + agent] &&
            next_ids[agent_base + occupant] == -1) {
          ++stack_top;
          stack_agents[agent_base + stack_top] = occupant;
          stack_offsets[agent_base + stack_top] = 0;
          descended = true;
        } else {
          completed_chain = true;
        }
        break;
      }

      if (completed_chain) {
        stack_top = -1;
      } else if (descended) {
        continue;
      } else if (stack_offsets[stack_index] >= 5) {
        const int64_t current_cell = current[agent_base + agent];
        occupied_next[cell_base + current_cell] = agent;
        next_ids[agent_base + agent] = current_cell;
        actions[agent_base + agent] = 0;
        --stack_top;
      }
    }
  }
}

__global__ void pibt_batched_kernel(
    const int64_t* current, const int64_t* candidates,
    const int64_t* preferences, const bool* valid, const int64_t* order,
    const int64_t* num_cells, int64_t* occupied_now, int64_t* occupied_next,
    int64_t* next_ids, int64_t* actions, int64_t* stack_agents,
    int64_t* stack_offsets, int64_t num_envs, int64_t num_agents,
    int64_t max_num_cells, bool populate_occupancy) {
  if (blockIdx.x >= num_envs || threadIdx.x != 0) return;
  pibt_environment(current, candidates, preferences, valid, order, num_cells,
      occupied_now, occupied_next, next_ids, actions, stack_agents, stack_offsets,
      num_envs, num_agents, max_num_cells, populate_occupancy, blockIdx.x);
}

__global__ void pibt_compact_prepare_kernel(
    const int64_t* __restrict__ current,
    const int64_t* __restrict__ candidates,
    const int64_t* __restrict__ preferences,
    const bool* __restrict__ valid,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ num_cells,
    int32_t* __restrict__ current_by_priority,
    int32_t* __restrict__ original_by_priority,
    int32_t* __restrict__ targets_by_preference,
    uint8_t* __restrict__ actions_by_preference,
    int32_t* __restrict__ occupied_now,
    int64_t num_envs,
    int64_t num_agents,
    int64_t max_num_cells) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_envs * num_agents;
  if (index >= total) return;

  const int64_t env = index / num_agents;
  const int64_t priority = index - env * num_agents;
  const int64_t agent_base = env * num_agents;
  const int64_t action_base = agent_base * 5;
  const int64_t original = order[agent_base + priority];
  const int64_t current_cell = current[agent_base + original];
  const int64_t cells = num_cells[env];

  current_by_priority[index] = static_cast<int32_t>(current_cell);
  original_by_priority[index] = static_cast<int32_t>(original);
  if (current_cell >= 0 && current_cell < cells) {
    occupied_now[env * max_num_cells + current_cell] =
        static_cast<int32_t>(priority);
  }

#pragma unroll
  for (int rank = 0; rank < 5; ++rank) {
    const int64_t packed_index = index * 5 + rank;
    const int64_t action = preferences[action_base + original * 5 + rank];
    int32_t target32 = -1;
    uint8_t action8 = 0;
    if (action >= 0 && action < 5) {
      action8 = static_cast<uint8_t>(action);
      const int64_t av_index = action_base + original * 5 + action;
      const int64_t target = candidates[av_index];
      if (valid[av_index] && target >= 0 && target < cells) {
        target32 = static_cast<int32_t>(target);
      }
    }
    targets_by_preference[packed_index] = target32;
    actions_by_preference[packed_index] = action8;
  }
}

__global__ void pibt_compact_output_kernel(
    const int32_t* __restrict__ original_by_priority,
    const int32_t* __restrict__ compact_next_ids,
    const uint8_t* __restrict__ compact_actions,
    int64_t* __restrict__ next_ids,
    int64_t* __restrict__ actions,
    int64_t total_agents,
    int64_t num_agents) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_agents) return;
  const int64_t original = original_by_priority[index];
  const int64_t agent_base = (index / num_agents) * num_agents;
  next_ids[agent_base + original] = compact_next_ids[index];
  actions[agent_base + original] = compact_actions[index];
}

__device__ int32_t pibt_component_root(
    int32_t* __restrict__ parent, int32_t agent) {
  int32_t next = parent[agent];
  while (next != agent) {
    agent = next;
    next = parent[agent];
  }
  return agent;
}

__device__ void pibt_component_union(
    int32_t* __restrict__ parent, int32_t first, int32_t second) {
  while (true) {
    first = pibt_component_root(parent, first);
    second = pibt_component_root(parent, second);
    if (first == second) return;
    const int32_t high = first > second ? first : second;
    const int32_t low = first > second ? second : first;
    if (atomicCAS(parent + high, high, low) == high) return;
  }
}

__device__ void pibt_component_connect_cell(
    int32_t* __restrict__ parent,
    int32_t* __restrict__ cell_owner,
    int32_t agent,
    int64_t cell_index) {
  const int32_t owner = atomicCAS(cell_owner + cell_index, -1, agent);
  if (owner != -1 && owner != agent) {
    pibt_component_union(parent, agent, owner);
  }
}

__global__ void pibt_component_parent_init_kernel(
    int32_t* __restrict__ parent, int64_t total_agents) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total_agents) parent[index] = static_cast<int32_t>(index);
}

__global__ void pibt_component_union_kernel(
    const int32_t* __restrict__ current,
    const int32_t* __restrict__ targets,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ cell_owner,
    int64_t total_agents,
    int64_t num_agents,
    int64_t max_num_cells) {
  const int64_t index64 =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index64 >= total_agents) return;
  const int32_t index = static_cast<int32_t>(index64);
  const int64_t env = index64 / num_agents;
  const int64_t cell_base = env * max_num_cells;

  pibt_component_connect_cell(
      parent, cell_owner, index, cell_base + current[index64]);
#pragma unroll
  for (int rank = 0; rank < 5; ++rank) {
    const int32_t target = targets[index64 * 5 + rank];
    if (target >= 0 && target != current[index64]) {
      pibt_component_connect_cell(
          parent, cell_owner, index, cell_base + target);
    }
  }
}

__global__ void pibt_component_key_kernel(
    int32_t* __restrict__ parent,
    int64_t* __restrict__ keys,
    int64_t total_agents) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_agents) return;
  const int32_t root = pibt_component_root(parent, static_cast<int32_t>(index));
  parent[index] = root;
  keys[index] =
      (static_cast<int64_t>(root) << 32) |
      static_cast<uint32_t>(index);
}

__global__ void pibt_component_boundaries_kernel(
    const int64_t* __restrict__ sorted_keys,
    int32_t* __restrict__ component_roots,
    int32_t* __restrict__ component_starts,
    int32_t* __restrict__ component_ends,
    int32_t* __restrict__ num_components,
    int64_t total_agents) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_agents) return;
  const int32_t root = static_cast<int32_t>(sorted_keys[index] >> 32);
  const bool begins_component =
      index == 0 ||
      static_cast<int32_t>(sorted_keys[index - 1] >> 32) != root;
  if (begins_component) {
    const int32_t slot = atomicAdd(num_components, 1);
    component_roots[slot] = root;
    component_starts[root] = static_cast<int32_t>(index);
    if (index > 0) {
      const int32_t previous_root =
          static_cast<int32_t>(sorted_keys[index - 1] >> 32);
      component_ends[previous_root] = static_cast<int32_t>(index);
    }
  }
  if (index + 1 == total_agents) {
    component_ends[root] = static_cast<int32_t>(total_agents);
  }
}

__global__ void pibt_batched_components_kernel(
    const int32_t* __restrict__ current,
    const int32_t* __restrict__ targets,
    const uint8_t* __restrict__ preferred_actions,
    const int64_t* __restrict__ sorted_keys,
    const int32_t* __restrict__ component_roots,
    const int32_t* __restrict__ component_starts,
    const int32_t* __restrict__ component_ends,
    int32_t* __restrict__ occupied_now,
    int32_t* __restrict__ occupied_next,
    int32_t* __restrict__ next_ids,
    uint8_t* __restrict__ actions,
    int32_t* __restrict__ stack_agents,
    uint8_t* __restrict__ stack_offsets,
    int64_t num_agents,
    int64_t max_num_cells,
    const int32_t* num_components) {
  const int32_t component = blockIdx.x;
  if (component >= *num_components || threadIdx.x != 0) return;
  const int32_t root = component_roots[component];
  const int32_t component_start = component_starts[root];
  const int32_t component_end = component_ends[root];
  const int32_t first_global_agent =
      static_cast<uint32_t>(sorted_keys[component_start]);
  const int64_t env = first_global_agent / num_agents;
  const int64_t agent_base = env * num_agents;
  const int64_t preference_base = agent_base * 5;
  const int64_t cell_base = env * max_num_cells;

  for (int32_t position = component_start;
       position < component_end;
       ++position) {
    const int32_t global_root =
        static_cast<uint32_t>(sorted_keys[position]);
    const int32_t local_root =
        static_cast<int32_t>(global_root - agent_base);
    if (next_ids[global_root] != -1) continue;
    int32_t stack_top = 0;
    stack_agents[component_start] = local_root;
    stack_offsets[component_start] = 0;

    while (stack_top >= 0) {
      const int64_t stack_index = component_start + stack_top;
      const int32_t agent = stack_agents[stack_index];
      bool descended = false;
      bool completed_chain = false;

      while (stack_offsets[stack_index] < 5) {
        const uint8_t rank = stack_offsets[stack_index]++;
        const int64_t preference_index =
            preference_base + static_cast<int64_t>(agent) * 5 + rank;
        const int32_t target = targets[preference_index];
        if (target < 0 || occupied_next[cell_base + target] != -1) continue;
        const int32_t occupant = occupied_now[cell_base + target];
        if (occupant != -1 &&
            next_ids[agent_base + occupant] == current[agent_base + agent]) {
          continue;
        }

        occupied_next[cell_base + target] = agent;
        next_ids[agent_base + agent] = target;
        actions[agent_base + agent] = preferred_actions[preference_index];
        if (occupant != -1 && target != current[agent_base + agent] &&
            next_ids[agent_base + occupant] == -1) {
          ++stack_top;
          stack_agents[component_start + stack_top] = occupant;
          stack_offsets[component_start + stack_top] = 0;
          descended = true;
        } else {
          completed_chain = true;
        }
        break;
      }

      if (completed_chain) {
        stack_top = -1;
      } else if (descended) {
        continue;
      } else if (stack_offsets[stack_index] >= 5) {
        const int32_t current_cell = current[agent_base + agent];
        occupied_next[cell_base + current_cell] = agent;
        next_ids[agent_base + agent] = current_cell;
        actions[agent_base + agent] = 0;
        --stack_top;
      }
    }
  }
}

void check_long_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
}
std::vector<torch::Tensor> cuda_pibt_resolve_batched(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells) {
  check_long_cuda(current_ids, "current_ids");
  check_long_cuda(candidate_ids, "candidate_ids");
  check_long_cuda(preferences, "preferences");
  check_long_cuda(order, "order");
  check_long_cuda(num_cells, "num_cells");
  TORCH_CHECK(valid.is_cuda() && valid.scalar_type() == torch::kBool &&
              valid.is_contiguous(), "valid must be contiguous CUDA bool");
  TORCH_CHECK(current_ids.dim() == 2, "current_ids must be [E,N]");
  TORCH_CHECK(candidate_ids.dim() == 3 && candidate_ids.size(2) == 5,
              "candidate_ids must be [E,N,5]");
  TORCH_CHECK(preferences.sizes() == candidate_ids.sizes(),
              "preferences must match candidate_ids");
  TORCH_CHECK(valid.sizes() == candidate_ids.sizes(),
              "valid must match candidate_ids");
  TORCH_CHECK(order.sizes() == current_ids.sizes(),
              "order must match current_ids");
  const auto num_envs = current_ids.size(0);
  const auto num_agents = current_ids.size(1);
  TORCH_CHECK(num_cells.numel() == num_envs,
              "num_cells must contain one value per environment");

  for (auto t : {candidate_ids, preferences, valid, order, num_cells})
    TORCH_CHECK(t.device() == current_ids.device(), "all inputs must share a CUDA device");
  TORCH_CHECK(candidate_ids.size(0)==num_envs && candidate_ids.size(1)==num_agents &&
              num_envs>0 && num_agents>0 && max_num_cells>0, "invalid batch shape");
  auto options = current_ids.options();
  auto occupied_now = torch::full({num_envs, max_num_cells}, -1, options);
  auto occupied_next = torch::full({num_envs, max_num_cells}, -1, options);
  auto next_ids = torch::full({num_envs, num_agents}, -1, options);
  auto actions = torch::zeros({num_envs, num_agents}, options);
  auto stack_agents = torch::empty({num_envs, num_agents}, options);
  auto stack_offsets = torch::empty({num_envs, num_agents}, options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  pibt_batched_kernel<<<num_envs, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
      current_ids.data_ptr<int64_t>(),
      candidate_ids.data_ptr<int64_t>(),
      preferences.data_ptr<int64_t>(),
      valid.data_ptr<bool>(),
      order.data_ptr<int64_t>(),
      num_cells.data_ptr<int64_t>(),
      occupied_now.data_ptr<int64_t>(),
      occupied_next.data_ptr<int64_t>(),
      next_ids.data_ptr<int64_t>(),
      actions.data_ptr<int64_t>(),
      stack_agents.data_ptr<int64_t>(),
      stack_offsets.data_ptr<int64_t>(),
      num_envs,
      num_agents,
      max_num_cells,
      true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {actions, next_ids};
}

std::vector<torch::Tensor> cuda_pibt_resolve_batched_components(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    torch::Tensor num_cells,
    int64_t max_num_cells) {
  check_long_cuda(current_ids, "current_ids");
  check_long_cuda(candidate_ids, "candidate_ids");
  check_long_cuda(preferences, "preferences");
  check_long_cuda(order, "order");
  check_long_cuda(num_cells, "num_cells");
  TORCH_CHECK(valid.is_cuda() && valid.scalar_type() == torch::kBool &&
              valid.is_contiguous(), "valid must be contiguous CUDA bool");
  TORCH_CHECK(current_ids.dim() == 2, "current_ids must be [E,N]");
  TORCH_CHECK(candidate_ids.dim() == 3 && candidate_ids.size(2) == 5,
              "candidate_ids must be [E,N,5]");
  TORCH_CHECK(preferences.sizes() == candidate_ids.sizes(),
              "preferences must match candidate_ids");
  TORCH_CHECK(valid.sizes() == candidate_ids.sizes(),
              "valid must match candidate_ids");
  TORCH_CHECK(order.sizes() == current_ids.sizes(),
              "order must match current_ids");
  const auto num_envs = current_ids.size(0);
  const auto num_agents = current_ids.size(1);
  const int64_t total_agents = num_envs * num_agents;
  TORCH_CHECK(num_cells.numel() == num_envs,
              "num_cells must contain one value per environment");
  TORCH_CHECK(
      total_agents <= std::numeric_limits<int32_t>::max(),
      "component resolver requires total agents <= int32 max");
  TORCH_CHECK(
      max_num_cells <= std::numeric_limits<int32_t>::max(),
      "component resolver requires max_num_cells <= int32 max");

  for (auto t : {candidate_ids, preferences, valid, order, num_cells})
    TORCH_CHECK(t.device() == current_ids.device(), "all inputs must share a CUDA device");
  TORCH_CHECK(candidate_ids.size(0)==num_envs && candidate_ids.size(1)==num_agents &&
              num_envs>0 && num_agents>0 && max_num_cells>0, "invalid batch shape");
  auto long_options = current_ids.options();
  auto int_options = current_ids.options().dtype(torch::kInt32);
  auto byte_options = current_ids.options().dtype(torch::kUInt8);
  auto current_by_priority = torch::empty({num_envs, num_agents}, int_options);
  auto original_by_priority = torch::empty({num_envs, num_agents}, int_options);
  auto targets_by_preference =
      torch::empty({num_envs, num_agents, 5}, int_options);
  auto actions_by_preference =
      torch::empty({num_envs, num_agents, 5}, byte_options);
  auto occupied_now = torch::full(
      {num_envs, max_num_cells}, -1, int_options);
  auto occupied_next = torch::full(
      {num_envs, max_num_cells}, -1, int_options);
  auto compact_next_ids = torch::full(
      {num_envs, num_agents}, -1, int_options);
  auto compact_actions = torch::zeros(
      {num_envs, num_agents}, byte_options);
  auto stack_agents = torch::empty({num_envs, num_agents}, int_options);
  auto stack_offsets = torch::empty({num_envs, num_agents}, byte_options);
  auto parent = torch::empty({total_agents}, int_options);
  auto cell_owner = torch::full(
      {num_envs, max_num_cells}, -1, int_options);
  auto component_keys = torch::empty({total_agents}, long_options);
  auto component_roots = torch::empty({total_agents}, int_options);
  auto component_starts = torch::empty({total_agents}, int_options);
  auto component_ends = torch::empty({total_agents}, int_options);
  auto num_components_tensor = torch::zeros({1}, int_options);
  auto next_ids = torch::empty({num_envs, num_agents}, long_options);
  auto actions = torch::empty({num_envs, num_agents}, long_options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(
      (total_agents + kThreads - 1) / kThreads);
  if (blocks > 0) {
    pibt_compact_prepare_kernel<<<
        blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        current_ids.data_ptr<int64_t>(),
        candidate_ids.data_ptr<int64_t>(),
        preferences.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        order.data_ptr<int64_t>(),
        num_cells.data_ptr<int64_t>(),
        current_by_priority.data_ptr<int32_t>(),
        original_by_priority.data_ptr<int32_t>(),
        targets_by_preference.data_ptr<int32_t>(),
        actions_by_preference.data_ptr<uint8_t>(),
        occupied_now.data_ptr<int32_t>(),
        num_envs,
        num_agents,
        max_num_cells);
    pibt_component_parent_init_kernel<<<
        blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        parent.data_ptr<int32_t>(), total_agents);
    pibt_component_union_kernel<<<
        blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        current_by_priority.data_ptr<int32_t>(),
        targets_by_preference.data_ptr<int32_t>(),
        parent.data_ptr<int32_t>(),
        cell_owner.data_ptr<int32_t>(),
        total_agents,
        num_agents,
        max_num_cells);
    pibt_component_key_kernel<<<
        blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        parent.data_ptr<int32_t>(),
        component_keys.data_ptr<int64_t>(),
        total_agents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto sorted_keys = std::get<0>(at::sort(component_keys, 0, false));
  if (blocks > 0) {
    pibt_component_boundaries_kernel<<<
        blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        sorted_keys.data_ptr<int64_t>(),
        component_roots.data_ptr<int32_t>(),
        component_starts.data_ptr<int32_t>(),
        component_ends.data_ptr<int32_t>(),
        num_components_tensor.data_ptr<int32_t>(),
        total_agents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // Device-side count: no mandatory CPU convergence/count readback.
  if (total_agents > 0) {
    pibt_batched_components_kernel<<<
        total_agents, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        current_by_priority.data_ptr<int32_t>(),
        targets_by_preference.data_ptr<int32_t>(),
        actions_by_preference.data_ptr<uint8_t>(),
        sorted_keys.data_ptr<int64_t>(),
        component_roots.data_ptr<int32_t>(),
        component_starts.data_ptr<int32_t>(),
        component_ends.data_ptr<int32_t>(),
        occupied_now.data_ptr<int32_t>(),
        occupied_next.data_ptr<int32_t>(),
        compact_next_ids.data_ptr<int32_t>(),
        compact_actions.data_ptr<uint8_t>(),
        stack_agents.data_ptr<int32_t>(),
        stack_offsets.data_ptr<uint8_t>(),
        num_agents,
        max_num_cells,
        num_components_tensor.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (blocks > 0) {
    pibt_compact_output_kernel<<<
        blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        original_by_priority.data_ptr<int32_t>(),
        compact_next_ids.data_ptr<int32_t>(),
        compact_actions.data_ptr<uint8_t>(),
        next_ids.data_ptr<int64_t>(),
        actions.data_ptr<int64_t>(),
        total_agents,
        num_agents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {actions, next_ids};
}
// Native policy compatibility: std::mt19937 plus libstdc++'s one-word
// uniform_real_distribution<float>. State is per environment, never global.
__global__ void native_mt_reset(int64_t* states, const int64_t* slots,
                               const int64_t* seeds, int64_t count, int64_t b) {
  int64_t i=blockIdx.x;
  if (i>=count || threadIdx.x!=0) return;
  int64_t slot=slots[i];
  if (slot<0 || slot>=b) return;
  auto s=states+slot*625;
  uint32_t x=static_cast<uint32_t>(seeds[i]);
  s[0]=x;
  for (int j=1;j<624;++j) { x=1812433253u*(x^(x>>30))+j; s[j]=x; }
  s[624]=624;
}

__device__ uint32_t native_mt_next(int64_t* s) {
  int index=static_cast<int>(s[624]);
  if (index==624) {
    for (int j=0;j<624;++j) {
      uint32_t y=(static_cast<uint32_t>(s[j])&0x80000000u) |
                 (static_cast<uint32_t>(s[(j+1)%624])&0x7fffffffu);
      s[j]=static_cast<uint32_t>(s[(j+397)%624])^(y>>1)^((y&1u)?0x9908b0dfu:0u);
    }
    index=0;
  }
  uint32_t y=static_cast<uint32_t>(s[index]); s[624]=index+1;
  y^=y>>11; y^=(y<<7)&0x9d2c5680u; y^=(y<<15)&0xefc60000u; y^=y>>18;
  return y;
}

__global__ void native_mt_ties(int64_t* states, const bool* valid,
                               const bool* active, float* ties, int64_t n) {
  int64_t env=blockIdx.x;
  if (threadIdx.x!=0 || !active[env]) return;
  auto s=states+env*625;
  const int graph_order[5]={3,4,2,1,0};
  for (int64_t i=0;i<n;++i) for (int k=0;k<5;++k) {
    int64_t offset=(env*n+i)*5+graph_order[k];
    if (valid[offset]) {
      // Integer-to-float rounding and clamp agree with generate_canonical.
      float value=static_cast<float>(native_mt_next(s))*0x1p-32f;
      ties[offset]=value>=1.0f?0x1.fffffep-1f:value;
    }
  }
}

void mt19937_reset(torch::Tensor state, torch::Tensor slots, torch::Tensor seeds) {
  check_long_cuda(state,"state"); check_long_cuda(slots,"slots"); check_long_cuda(seeds,"seeds");
  TORCH_CHECK(state.dim()==2 && state.size(1)==625,"state must be [B,625]");
  TORCH_CHECK(slots.dim()==1 && seeds.sizes()==slots.sizes(),"slots/seeds must be [K]");
  TORCH_CHECK(slots.device()==state.device() && seeds.device()==state.device(),"device mismatch");
  c10::cuda::CUDAGuard guard(state.device());
  if (slots.numel()) {
    native_mt_reset<<<slots.numel(),1,0,c10::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<int64_t>(),slots.data_ptr<int64_t>(),seeds.data_ptr<int64_t>(),slots.numel(),state.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

torch::Tensor mt19937_ties(torch::Tensor state, torch::Tensor valid, torch::Tensor active) {
  check_long_cuda(state,"state");
  TORCH_CHECK(state.dim()==2 && state.size(1)==625,"state must be [B,625]");
  TORCH_CHECK(valid.is_cuda() && valid.scalar_type()==torch::kBool && valid.is_contiguous() &&
              valid.dim()==3 && valid.size(0)==state.size(0) && valid.size(2)==5,"valid must be CUDA bool[B,N,5]");
  TORCH_CHECK(active.is_cuda() && active.scalar_type()==torch::kBool && active.is_contiguous() &&
              active.dim()==1 && active.size(0)==state.size(0),"active must be CUDA bool[B]");
  TORCH_CHECK(valid.device()==state.device() && active.device()==state.device(),"device mismatch");
  c10::cuda::CUDAGuard guard(state.device());
  auto ties=torch::zeros(valid.sizes(),state.options().dtype(torch::kFloat32));
  if (state.size(0)) {
    native_mt_ties<<<state.size(0),1,0,c10::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<int64_t>(),valid.data_ptr<bool>(),active.data_ptr<bool>(),ties.data_ptr<float>(),valid.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return ties;
}

#include "pibt_rse.cuh"
#include "native_priority.cuh"

__global__ void native_order_kernel(const float* p, int64_t* order, int64_t n) {
  const int64_t base=blockIdx.x*n;
  dmm_native_order::sort(order+base,n,p+base);
}

torch::Tensor native_priority_order(torch::Tensor priorities) {
  TORCH_CHECK(priorities.is_cuda() && priorities.is_contiguous() && priorities.dim()==2 &&
              priorities.scalar_type()==torch::kFloat32 && priorities.size(1)>0,
              "priorities must be contiguous CUDA float32 [B,N]");
  c10::cuda::CUDAGuard guard(priorities.device());
  auto order=torch::empty(priorities.sizes(),priorities.options().dtype(torch::kInt64));
  if (priorities.size(0)) {
    native_order_kernel<<<priorities.size(0),1,0,c10::cuda::getCurrentCUDAStream()>>>(
        priorities.data_ptr<float>(),order.data_ptr<int64_t>(),priorities.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return order;
}

__global__ void native_priority_init_kernel(const int64_t* distances, float* out,
                                           int64_t n, int64_t free_cells) {
  int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) out[i]=__fdiv_rn(static_cast<float>(distances[i]<0?free_cells:distances[i]),10000.0f);
}
torch::Tensor native_priority_init(torch::Tensor distances, int64_t free_cells) {
  check_long_cuda(distances,"distances");
  TORCH_CHECK(distances.dim()==1 && free_cells>0,"distances must be [N], free_cells positive");
  c10::cuda::CUDAGuard guard(distances.device());
  auto out=torch::empty(distances.sizes(),distances.options().dtype(torch::kFloat32));
  if (distances.numel()) {
    native_priority_init_kernel<<<(distances.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
        distances.data_ptr<int64_t>(),out.data_ptr<float>(),distances.numel(),free_cells);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("native_priority_order", &native_priority_order);
 m.def("native_priority_init", &native_priority_init);
 m.def("rse", &pibt_rse_allocating);
 m.def("rse_workspace", &pibt_rse);
 m.def("mt19937_reset", &mt19937_reset);
 m.def("mt19937_ties", &mt19937_ties);
 m.def("sequential", &cuda_pibt_resolve_batched);
 m.def("components", &cuda_pibt_resolve_batched_components);
}
