#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

__global__ void pibt_kernel(
    const int64_t* current,
    const int64_t* candidates,
    const int64_t* preferences,
    const bool* valid,
    const int64_t* order,
    int64_t* occupied_now,
    int64_t* occupied_next,
    int64_t* next_ids,
    int64_t* actions,
    int64_t* stack_agents,
    int64_t* stack_offsets,
    int64_t num_agents,
    int64_t num_cells) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  for (int64_t agent = 0; agent < num_agents; ++agent) {
    occupied_now[current[agent]] = agent;
  }

  for (int64_t root_index = 0; root_index < num_agents; ++root_index) {
    const int64_t root = order[root_index];
    if (next_ids[root] != -1) continue;
    int64_t stack_top = 0;
    stack_agents[0] = root;
    stack_offsets[0] = 0;

    while (stack_top >= 0) {
      const int64_t agent = stack_agents[stack_top];
      bool descended = false;
      bool completed_chain = false;

      while (stack_offsets[stack_top] < 5) {
        const int64_t rank = stack_offsets[stack_top]++;
        const int64_t action = preferences[agent * 5 + rank];
        if (action < 0 || action >= 5 || !valid[agent * 5 + action]) continue;
        const int64_t target = candidates[agent * 5 + action];
        if (target < 0 || target >= num_cells || occupied_next[target] != -1) continue;
        const int64_t occupant = occupied_now[target];
        if (occupant != -1 && next_ids[occupant] == current[agent]) continue;

        occupied_next[target] = agent;
        next_ids[agent] = target;
        actions[agent] = action;
        if (occupant != -1 && target != current[agent] &&
            next_ids[occupant] == -1) {
          ++stack_top;
          stack_agents[stack_top] = occupant;
          stack_offsets[stack_top] = 0;
          descended = true;
        } else {
          // A successful recursive child makes every ancestor successful.
          completed_chain = true;
        }
        break;
      }

      if (completed_chain) {
        stack_top = -1;
      } else if (descended) {
        continue;
      } else if (stack_offsets[stack_top] >= 5) {
        occupied_next[current[agent]] = agent;
        next_ids[agent] = current[agent];
        actions[agent] = 0;
        --stack_top;
      }
    }
  }
}

// One exact sequential PIBT state machine per CUDA block.  The environments
// are independent, so a single control thread per block gives us exact PIBT
// semantics without thousands of host-side extension calls per rollout.
__global__ void pibt_batched_kernel(
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
    bool populate_occupancy) {
  const int64_t env = blockIdx.x;
  if (env >= num_envs || threadIdx.x != 0) return;

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

// Populate the current occupancy table independently of the exact PIBT state
// machine.  Valid MAPF states have at most one agent in each cell, so these
// stores are race-free and produce the same table as the serial loop above.
__global__ void pibt_batched_occupancy_kernel(
    const int64_t* current,
    const int64_t* num_cells,
    int64_t* occupied_now,
    int64_t num_envs,
    int64_t num_agents,
    int64_t max_num_cells) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_envs * num_agents;
  if (index >= total) return;

  const int64_t env = index / num_agents;
  const int64_t agent = index - env * num_agents;
  const int64_t cell = current[index];
  if (cell >= 0 && cell < num_cells[env]) {
    occupied_now[env * max_num_cells + cell] = agent;
  }
}

// Rename agents by priority rank and pack the immutable resolver inputs.  The
// renaming is a bijection, so it cannot affect PIBT decisions; it makes the
// common root traversal sequential in memory while recursion still addresses
// the same occupants through their renamed ids.
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

// Same ordered PIBT state machine as pibt_batched_kernel, operating on the
// compact priority-renamed representation produced above.
__global__ __launch_bounds__(1) void pibt_batched_compact_kernel(
    const int32_t* __restrict__ current,
    const int32_t* __restrict__ targets,
    const uint8_t* __restrict__ preferred_actions,
    int32_t* __restrict__ occupied_now,
    int32_t* __restrict__ occupied_next,
    int32_t* __restrict__ next_ids,
    uint8_t* __restrict__ actions,
    int32_t* __restrict__ stack_agents,
    uint8_t* __restrict__ stack_offsets,
    int64_t num_envs,
    int64_t num_agents,
    int64_t max_num_cells) {
  const int64_t env = blockIdx.x;
  if (env >= num_envs || threadIdx.x != 0) return;

  const int64_t agent_base = env * num_agents;
  const int64_t preference_base = agent_base * 5;
  const int64_t cell_base = env * max_num_cells;

  for (int32_t root = 0; root < static_cast<int32_t>(num_agents); ++root) {
    if (next_ids[agent_base + root] != -1) continue;
    int32_t stack_top = 0;
    stack_agents[agent_base] = root;
    stack_offsets[agent_base] = 0;

    while (stack_top >= 0) {
      const int64_t stack_index = agent_base + stack_top;
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
        const int32_t current_cell = current[agent_base + agent];
        occupied_next[cell_base + current_cell] = agent;
        next_ids[agent_base + agent] = current_cell;
        actions[agent_base + agent] = 0;
        --stack_top;
      }
    }
  }
}

void pibt_batched_compact_cpu(
    const int32_t* __restrict__ current,
    const int32_t* __restrict__ targets,
    const uint8_t* __restrict__ preferred_actions,
    int32_t* __restrict__ occupied_now,
    int32_t* __restrict__ occupied_next,
    int32_t* __restrict__ next_ids,
    uint8_t* __restrict__ actions,
    int32_t* __restrict__ stack_agents,
    uint8_t* __restrict__ stack_offsets,
    int64_t num_envs,
    int64_t num_agents,
    int64_t max_num_cells) {
  for (int64_t env = 0; env < num_envs; ++env) {
    const int64_t agent_base = env * num_agents;
    const int64_t preference_base = agent_base * 5;
    const int64_t cell_base = env * max_num_cells;

    for (int32_t root = 0; root < static_cast<int32_t>(num_agents); ++root) {
      if (next_ids[agent_base + root] != -1) continue;
      int32_t stack_top = 0;
      stack_agents[agent_base] = root;
      stack_offsets[agent_base] = 0;

      while (stack_top >= 0) {
        const int64_t stack_index = agent_base + stack_top;
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
          const int32_t current_cell = current[agent_base + agent];
          occupied_next[cell_base + current_cell] = agent;
          next_ids[agent_base + agent] = current_cell;
          actions[agent_base + agent] = 0;
          --stack_top;
        }
      }
    }
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

// Agents are connected when any valid candidate cell overlaps.  The current
// cell is included explicitly because exact PIBT may fall back to waiting even
// when action zero was externally forbidden.
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

// Each block owns a component whose complete candidate-cell set is disjoint
// from every other component.  A single lane preserves exact ordered PIBT
// semantics within that component; the GPU schedules all components in
// parallel to hide branch and random-access latency.
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
    int32_t num_components) {
  const int32_t component = blockIdx.x;
  if (component >= num_components || threadIdx.x != 0) return;
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

// Exact PIBT with one warp per independent environment.  Lane zero retains
// sole ownership of all state transitions.  The warp only evaluates the
// remaining preference ranks concurrently; __ffs selects the first acceptable
// rank, exactly matching the sequential resolver's iteration order.
__global__ void pibt_batched_warp_kernel(
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
    int64_t max_num_cells) {
  const int64_t env = blockIdx.x;
  const int lane = threadIdx.x;
  if (env >= num_envs || lane >= 32) return;

  constexpr unsigned kWarpMask = 0xffffffffu;
  const int64_t agent_base = env * num_agents;
  const int64_t action_base = agent_base * 5;
  const int64_t cell_base = env * max_num_cells;
  const int64_t cells = num_cells[env];

  __shared__ int64_t shared_stack_top;
  __shared__ int64_t shared_agent;
  __shared__ int64_t shared_offset;

  for (int64_t root_index = 0; root_index < num_agents; ++root_index) {
    const int64_t root = order[agent_base + root_index];
    if (next_ids[agent_base + root] != -1) continue;

    if (lane == 0) {
      shared_stack_top = 0;
      stack_agents[agent_base] = root;
      stack_offsets[agent_base] = 0;
    }
    __syncwarp(kWarpMask);

    while (shared_stack_top >= 0) {
      if (lane == 0) {
        const int64_t stack_index = agent_base + shared_stack_top;
        shared_agent = stack_agents[stack_index];
        shared_offset = stack_offsets[stack_index];
      }
      __syncwarp(kWarpMask);

      const int64_t agent = shared_agent;
      const int64_t offset = shared_offset;
      const int64_t rank = offset + lane;
      bool acceptable = false;

      if (rank < 5) {
        const int64_t pref_index = action_base + agent * 5 + rank;
        const int64_t action = preferences[pref_index];
        if (action >= 0 && action < 5) {
          const int64_t av_index = action_base + agent * 5 + action;
          if (valid[av_index]) {
            const int64_t target = candidates[av_index];
            if (target >= 0 && target < cells &&
                occupied_next[cell_base + target] == -1) {
              const int64_t occupant = occupied_now[cell_base + target];
              acceptable =
                  occupant == -1 ||
                  next_ids[agent_base + occupant] != current[agent_base + agent];
            }
          }
        }
      }

      const unsigned acceptable_mask = __ballot_sync(kWarpMask, acceptable);
      const int chosen_lane = __ffs(acceptable_mask) - 1;

      if (lane == 0) {
        const int64_t stack_index = agent_base + shared_stack_top;
        if (chosen_lane >= 0) {
          const int64_t chosen_rank = offset + chosen_lane;
          stack_offsets[stack_index] = chosen_rank + 1;
          const int64_t pref_index = action_base + agent * 5 + chosen_rank;
          const int64_t action = preferences[pref_index];
          const int64_t av_index = action_base + agent * 5 + action;
          const int64_t target = candidates[av_index];
          const int64_t occupant = occupied_now[cell_base + target];

          occupied_next[cell_base + target] = agent;
          next_ids[agent_base + agent] = target;
          actions[agent_base + agent] = action;
          if (occupant != -1 && target != current[agent_base + agent] &&
              next_ids[agent_base + occupant] == -1) {
            ++shared_stack_top;
            stack_agents[agent_base + shared_stack_top] = occupant;
            stack_offsets[agent_base + shared_stack_top] = 0;
          } else {
            // A successful recursive child makes every ancestor successful.
            shared_stack_top = -1;
          }
        } else {
          stack_offsets[stack_index] = 5;
          const int64_t current_cell = current[agent_base + agent];
          occupied_next[cell_base + current_cell] = agent;
          next_ids[agent_base + agent] = current_cell;
          actions[agent_base + agent] = 0;
          --shared_stack_top;
        }
      }
      __syncwarp(kWarpMask);
    }
  }
}

void check_long_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> cuda_pibt_resolve(
    torch::Tensor current_ids,
    torch::Tensor candidate_ids,
    torch::Tensor preferences,
    torch::Tensor valid,
    torch::Tensor order,
    int64_t num_cells) {
  check_long_cuda(current_ids, "current_ids");
  check_long_cuda(candidate_ids, "candidate_ids");
  check_long_cuda(preferences, "preferences");
  check_long_cuda(order, "order");
  TORCH_CHECK(valid.is_cuda() && valid.scalar_type() == torch::kBool &&
              valid.is_contiguous(), "valid must be contiguous CUDA bool");
  TORCH_CHECK(candidate_ids.dim() == 2 && candidate_ids.size(1) == 5,
              "candidate_ids must be [N,5]");
  const auto num_agents = current_ids.numel();
  auto options = current_ids.options();
  // Tensor fill kernels initialize large MovingAI maps in parallel; the PIBT
  // control kernel itself remains a compact exact sequential state machine.
  auto occupied_now = torch::full({num_cells}, -1, options);
  auto occupied_next = torch::full({num_cells}, -1, options);
  auto next_ids = torch::full({num_agents}, -1, options);
  auto actions = torch::zeros({num_agents}, options);
  auto stack_agents = torch::empty({num_agents}, options);
  auto stack_offsets = torch::empty({num_agents}, options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  pibt_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      current_ids.data_ptr<int64_t>(),
      candidate_ids.data_ptr<int64_t>(),
      preferences.data_ptr<int64_t>(),
      valid.data_ptr<bool>(),
      order.data_ptr<int64_t>(),
      occupied_now.data_ptr<int64_t>(),
      occupied_next.data_ptr<int64_t>(),
      next_ids.data_ptr<int64_t>(),
      actions.data_ptr<int64_t>(),
      stack_agents.data_ptr<int64_t>(),
      stack_offsets.data_ptr<int64_t>(),
      num_agents,
      num_cells);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {actions, next_ids};
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

  auto options = current_ids.options();
  auto occupied_now = torch::full({num_envs, max_num_cells}, -1, options);
  auto occupied_next = torch::full({num_envs, max_num_cells}, -1, options);
  auto next_ids = torch::full({num_envs, num_agents}, -1, options);
  auto actions = torch::zeros({num_envs, num_agents}, options);
  auto stack_agents = torch::empty({num_envs, num_agents}, options);
  auto stack_offsets = torch::empty({num_envs, num_agents}, options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  pibt_batched_kernel<<<num_envs, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
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

std::vector<torch::Tensor> cuda_pibt_resolve_batched_parallel_occupancy(
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

  auto options = current_ids.options();
  auto occupied_now = torch::full({num_envs, max_num_cells}, -1, options);
  auto occupied_next = torch::full({num_envs, max_num_cells}, -1, options);
  auto next_ids = torch::full({num_envs, num_agents}, -1, options);
  auto actions = torch::zeros({num_envs, num_agents}, options);
  auto stack_agents = torch::empty({num_envs, num_agents}, options);
  auto stack_offsets = torch::empty({num_envs, num_agents}, options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  constexpr int kOccupancyThreads = 256;
  const int64_t total_agents = num_envs * num_agents;
  const int occupancy_blocks = static_cast<int>(
      (total_agents + kOccupancyThreads - 1) / kOccupancyThreads);
  if (occupancy_blocks > 0) {
    pibt_batched_occupancy_kernel<<<
        occupancy_blocks,
        kOccupancyThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        current_ids.data_ptr<int64_t>(),
        num_cells.data_ptr<int64_t>(),
        occupied_now.data_ptr<int64_t>(),
        num_envs,
        num_agents,
        max_num_cells);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  pibt_batched_kernel<<<num_envs, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
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
      false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {actions, next_ids};
}

std::vector<torch::Tensor> cuda_pibt_resolve_batched_compact(
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
  TORCH_CHECK(
      num_agents <= std::numeric_limits<int32_t>::max(),
      "compact resolver requires num_agents <= int32 max");
  TORCH_CHECK(
      max_num_cells <= std::numeric_limits<int32_t>::max(),
      "compact resolver requires max_num_cells <= int32 max");

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
  auto next_ids = torch::empty({num_envs, num_agents}, long_options);
  auto actions = torch::empty({num_envs, num_agents}, long_options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  constexpr int kThreads = 256;
  const int64_t total_agents = num_envs * num_agents;
  const int blocks = static_cast<int>(
      (total_agents + kThreads - 1) / kThreads);
  if (blocks > 0) {
    pibt_compact_prepare_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  pibt_batched_compact_kernel<<<
      num_envs, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      current_by_priority.data_ptr<int32_t>(),
      targets_by_preference.data_ptr<int32_t>(),
      actions_by_preference.data_ptr<uint8_t>(),
      occupied_now.data_ptr<int32_t>(),
      occupied_next.data_ptr<int32_t>(),
      compact_next_ids.data_ptr<int32_t>(),
      compact_actions.data_ptr<uint8_t>(),
      stack_agents.data_ptr<int32_t>(),
      stack_offsets.data_ptr<uint8_t>(),
      num_envs,
      num_agents,
      max_num_cells);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (blocks > 0) {
    pibt_compact_output_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        parent.data_ptr<int32_t>(), total_agents);
    pibt_component_union_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        current_by_priority.data_ptr<int32_t>(),
        targets_by_preference.data_ptr<int32_t>(),
        parent.data_ptr<int32_t>(),
        cell_owner.data_ptr<int32_t>(),
        total_agents,
        num_agents,
        max_num_cells);
    pibt_component_key_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        parent.data_ptr<int32_t>(),
        component_keys.data_ptr<int64_t>(),
        total_agents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto sorted_keys = std::get<0>(at::sort(component_keys, 0, false));
  if (blocks > 0) {
    pibt_component_boundaries_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        sorted_keys.data_ptr<int64_t>(),
        component_roots.data_ptr<int32_t>(),
        component_starts.data_ptr<int32_t>(),
        component_ends.data_ptr<int32_t>(),
        num_components_tensor.data_ptr<int32_t>(),
        total_agents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int32_t num_components =
      num_components_tensor.cpu().item<int32_t>();
  if (num_components > 0) {
    pibt_batched_components_kernel<<<
        num_components, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
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
        num_components);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (blocks > 0) {
    pibt_compact_output_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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

std::vector<torch::Tensor> cuda_pibt_resolve_batched_hybrid(
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
  TORCH_CHECK(
      num_agents <= std::numeric_limits<int32_t>::max(),
      "hybrid resolver requires num_agents <= int32 max");
  TORCH_CHECK(
      max_num_cells <= std::numeric_limits<int32_t>::max(),
      "hybrid resolver requires max_num_cells <= int32 max");

  auto long_options = current_ids.options();
  auto gpu_int_options = current_ids.options().dtype(torch::kInt32);
  auto gpu_byte_options = current_ids.options().dtype(torch::kUInt8);
  auto cpu_int_options = torch::TensorOptions()
      .device(torch::kCPU).dtype(torch::kInt32).pinned_memory(true);
  auto cpu_byte_options = torch::TensorOptions()
      .device(torch::kCPU).dtype(torch::kUInt8).pinned_memory(true);

  auto gpu_current = torch::empty({num_envs, num_agents}, gpu_int_options);
  auto original_by_priority =
      torch::empty({num_envs, num_agents}, gpu_int_options);
  auto gpu_targets =
      torch::empty({num_envs, num_agents, 5}, gpu_int_options);
  auto gpu_preferred_actions =
      torch::empty({num_envs, num_agents, 5}, gpu_byte_options);
  auto gpu_occupied_now = torch::full(
      {num_envs, max_num_cells}, -1, gpu_int_options);
  auto gpu_next_ids = torch::empty({num_envs, num_agents}, gpu_int_options);
  auto gpu_actions = torch::empty({num_envs, num_agents}, gpu_byte_options);
  auto next_ids = torch::empty({num_envs, num_agents}, long_options);
  auto actions = torch::empty({num_envs, num_agents}, long_options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  constexpr int kThreads = 256;
  const int64_t total_agents = num_envs * num_agents;
  const int blocks = static_cast<int>(
      (total_agents + kThreads - 1) / kThreads);
  if (blocks > 0) {
    pibt_compact_prepare_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        current_ids.data_ptr<int64_t>(),
        candidate_ids.data_ptr<int64_t>(),
        preferences.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        order.data_ptr<int64_t>(),
        num_cells.data_ptr<int64_t>(),
        gpu_current.data_ptr<int32_t>(),
        original_by_priority.data_ptr<int32_t>(),
        gpu_targets.data_ptr<int32_t>(),
        gpu_preferred_actions.data_ptr<uint8_t>(),
        gpu_occupied_now.data_ptr<int32_t>(),
        num_envs,
        num_agents,
        max_num_cells);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto cpu_current = torch::empty({num_envs, num_agents}, cpu_int_options);
  auto cpu_targets =
      torch::empty({num_envs, num_agents, 5}, cpu_int_options);
  auto cpu_preferred_actions =
      torch::empty({num_envs, num_agents, 5}, cpu_byte_options);
  auto cpu_occupied_now = torch::empty(
      {num_envs, max_num_cells}, cpu_int_options);
  cpu_current.copy_(gpu_current);
  cpu_targets.copy_(gpu_targets);
  cpu_preferred_actions.copy_(gpu_preferred_actions);
  cpu_occupied_now.copy_(gpu_occupied_now);

  auto cpu_occupied_next = torch::full(
      {num_envs, max_num_cells}, -1, cpu_int_options);
  auto cpu_next_ids = torch::full(
      {num_envs, num_agents}, -1, cpu_int_options);
  auto cpu_actions = torch::zeros(
      {num_envs, num_agents}, cpu_byte_options);
  auto cpu_stack_agents = torch::empty(
      {num_envs, num_agents}, cpu_int_options);
  auto cpu_stack_offsets = torch::empty(
      {num_envs, num_agents}, cpu_byte_options);

  pibt_batched_compact_cpu(
      cpu_current.data_ptr<int32_t>(),
      cpu_targets.data_ptr<int32_t>(),
      cpu_preferred_actions.data_ptr<uint8_t>(),
      cpu_occupied_now.data_ptr<int32_t>(),
      cpu_occupied_next.data_ptr<int32_t>(),
      cpu_next_ids.data_ptr<int32_t>(),
      cpu_actions.data_ptr<uint8_t>(),
      cpu_stack_agents.data_ptr<int32_t>(),
      cpu_stack_offsets.data_ptr<uint8_t>(),
      num_envs,
      num_agents,
      max_num_cells);

  gpu_next_ids.copy_(cpu_next_ids);
  gpu_actions.copy_(cpu_actions);
  if (blocks > 0) {
    pibt_compact_output_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        original_by_priority.data_ptr<int32_t>(),
        gpu_next_ids.data_ptr<int32_t>(),
        gpu_actions.data_ptr<uint8_t>(),
        next_ids.data_ptr<int64_t>(),
        actions.data_ptr<int64_t>(),
        total_agents,
        num_agents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {actions, next_ids};
}

std::vector<torch::Tensor> cuda_pibt_resolve_batched_warp(
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

  auto options = current_ids.options();
  auto occupied_now = torch::full({num_envs, max_num_cells}, -1, options);
  auto occupied_next = torch::full({num_envs, max_num_cells}, -1, options);
  auto next_ids = torch::full({num_envs, num_agents}, -1, options);
  auto actions = torch::zeros({num_envs, num_agents}, options);
  auto stack_agents = torch::empty({num_envs, num_agents}, options);
  auto stack_offsets = torch::empty({num_envs, num_agents}, options);

  c10::cuda::CUDAGuard guard(current_ids.device());
  constexpr int kOccupancyThreads = 256;
  const int64_t total_agents = num_envs * num_agents;
  const int occupancy_blocks = static_cast<int>(
      (total_agents + kOccupancyThreads - 1) / kOccupancyThreads);
  if (occupancy_blocks > 0) {
    pibt_batched_occupancy_kernel<<<
        occupancy_blocks,
        kOccupancyThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        current_ids.data_ptr<int64_t>(),
        num_cells.data_ptr<int64_t>(),
        occupied_now.data_ptr<int64_t>(),
        num_envs,
        num_agents,
        max_num_cells);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  pibt_batched_warp_kernel<<<
      num_envs, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
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
      max_num_cells);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {actions, next_ids};
}
