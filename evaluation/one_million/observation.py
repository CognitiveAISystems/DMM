import torch
from evaluation.one_million.profiling import NullProfiler

class GPUObservationGenerator:
    def __init__(self, width, height, grid, cfg, profiler=None, cache_radius=None,
                 on_target="nothing"):
        self.profiler = profiler or NullProfiler()
        self.width = width
        self.height = height
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.on_target = on_target

        obs_radius = cfg.cost2go_radius
        if cache_radius is None:
            cache_radius = 3 * obs_radius
        self.cache_radius = cache_radius
        self.cache_enabled = cache_radius > obs_radius

        self.obstacles_bool = torch.as_tensor(grid, dtype=torch.uint8, device=self.device) \
            .reshape((self.height, self.width)).bool()

        self.action_history = None
        self.pos_t = None
        self.goals_t = None
        self.current_goal_idx = None
        self.active_goal_mask = None

        from pogema_gpu.kernels.cuda_bfs import is_available
        if not is_available():
            raise RuntimeError(
                "CUDA BFS kernel failed to compile or pass smoke test. "
                "Ensure CUDA toolkit is installed and CUDA_HOME is set."
            )

        self._build_vocab()

        self.inf_dist = self.height + self.width  # larger than any manhattan distance
        self.pad_tensor_val = torch.tensor(self.pad_token, dtype=torch.long, device=self.device)

    def _build_vocab(self):
        limit = self.cfg.cost2go_value_limit
        coord_range = list(range(-limit, limit + 1)) + [-limit * 4, -limit * 2, limit * 2]
        actions_range = ['n', 'w', 'u', 'd', 'l', 'r']
        next_action_range = [f"{i:04b}" for i in range(16)]

        int_vocab = {}
        str_vocab = {}
        idx = 0
        for token in coord_range:
            int_vocab[token] = idx
            idx += 1
        for token in actions_range:
            str_vocab[token] = idx
            idx += 1
        for token in next_action_range:
            str_vocab[token] = idx
            idx += 1
        str_vocab["!"] = idx
        self.pad_token = idx

        self.coord_offset = limit * 4
        self.coord_lookup = torch.full((limit * 8 + 1,), self.pad_token, dtype=torch.long, device=self.device)
        for val in coord_range:
            self.coord_lookup[val + self.coord_offset] = int_vocab[val]

        n_tok = str_vocab['n']
        self.env_act_to_token = torch.tensor([
            str_vocab['w'],
            str_vocab['u'],
            str_vocab['d'],
            str_vocab['l'],
            str_vocab['r'],
            n_tok
        ], dtype=torch.long, device=self.device)

        self.next_action_base = str_vocab['0000']
        self.default_hist_token = n_tok


    def create_agents(self, positions, goals, current_goal_idx=None, active_goal_mask=None):
        N = len(positions)
        self.action_history = torch.full(
            (N, self.cfg.num_previous_actions),
            self.default_hist_token,
            dtype=torch.long,
            device=self.device
        )
        self.pos_t = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        self.goals_t = torch.as_tensor(goals, dtype=torch.long, device=self.device)

        if self.cache_enabled:
            cache_side = 2 * self.cache_radius + 1
            self._cache_windows = torch.full(
                (N, cache_side * cache_side), -1, dtype=torch.int16, device=self.device)
            self._cache_center = torch.zeros(N, 2, dtype=torch.long, device=self.device)
            self._cache_valid = torch.zeros(N, dtype=torch.bool, device=self.device)
            self.current_goal_idx = current_goal_idx
            self.active_goal_mask = active_goal_mask

    def update_agents(self, positions, goals, last_actions):
        if self.cache_enabled and self.on_target == 'restart' and self.active_goal_mask.any():
            self._cache_valid[self.active_goal_mask] = False

        self.pos_t = positions
        self.goals_t = goals

        if last_actions is not None:
            la_tensor = torch.as_tensor(last_actions, dtype=torch.long, device=self.device)
            la_tensor = torch.where((la_tensor >= 0) & (la_tensor <= 4), la_tensor, 5)
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1] = self.env_act_to_token[la_tensor]

    def generate_observations(self):
        N = self.pos_t.size(0)
        cost2go_tokens, next_action_tokens = self._compute_cost2go_and_actions(self.pos_t, self.goals_t)
        with self.profiler.time("generate_observations/neighbors"):
            return self._get_agents_info(N, self.pos_t, self.goals_t, cost2go_tokens, next_action_tokens)

    def _compute_cost2go_and_actions(self, pos_t, goals_t):
        if not self.cache_enabled:
            return self._compute_cost2go_uncached(pos_t, goals_t)

        limit = self.cfg.cost2go_value_limit
        obs_radius = self.cfg.cost2go_radius
        cache_radius = self.cache_radius

        with self.profiler.time("generate_observations/cache_check"):
            margin = cache_radius - obs_radius
            delta = (pos_t - self._cache_center).abs()
            valid = self._cache_valid & (delta.max(dim=1).values <= margin)
            miss_mask = ~valid

        if miss_mask.any():
            miss_idx = miss_mask.nonzero(as_tuple=True)[0]
            with self.profiler.time("generate_observations/bfs"):
                from pogema_gpu.kernels.cuda_bfs import raw_bfs_cost2go
                chunk = int(getattr(self.cfg, "bfs_chunk_size", 0) or 0)
                if chunk <= 0:
                    chunk = miss_idx.numel()
                for start in range(0, miss_idx.numel(), chunk):
                    batch = miss_idx[start:start + chunk]
                    # Bound each launch to cap its per-launch workspace.
                    raw_windows = raw_bfs_cost2go(
                        self.obstacles_bool,
                        pos_t[batch], goals_t[batch],
                        self.height, self.width, cache_radius)
                    self._cache_windows[batch] = raw_windows
                    self._cache_center[batch] = pos_t[batch]
                    self._cache_valid[batch] = True

        with self.profiler.time("generate_observations/extract"):
            from pogema_gpu.kernels.cuda_bfs import extract_and_normalize_cuda
            windows, mask_codes = extract_and_normalize_cuda(
                    self._cache_windows, pos_t, self._cache_center,
                    limit, obs_radius, cache_radius
                )

        with self.profiler.time("generate_observations/token_lookup"):
            idx = (windows.long() + self.coord_offset).clamp_(0, len(self.coord_lookup) - 1)
            cost2go_tokens = self.coord_lookup[idx]
            next_action_tokens = mask_codes.long() + self.next_action_base

        return cost2go_tokens, next_action_tokens

    def _compute_cost2go_uncached(self, pos_t, goals_t):
        limit = self.cfg.cost2go_value_limit
        radius = self.cfg.cost2go_radius
        from pogema_gpu.kernels.cuda_bfs import fused_bfs_cost2go

        with self.profiler.time("generate_observations/bfs"):
            N = pos_t.size(0)
            chunk = int(getattr(self.cfg, "bfs_chunk_size", 0) or 0)
            if chunk <= 0 or chunk >= N:
                windows, mask_codes = fused_bfs_cost2go(
                    self.obstacles_bool, pos_t, goals_t,
                    self.height, self.width, radius, limit)
            else:
                side = 2 * radius + 1
                windows = torch.empty(
                    (N, side * side), dtype=torch.int16, device=pos_t.device
                )
                mask_codes = torch.empty((N,), dtype=torch.int32, device=pos_t.device)
                for start in range(0, N, chunk):
                    end = min(start + chunk, N)
                    current_windows, current_codes = fused_bfs_cost2go(
                        self.obstacles_bool,
                        pos_t[start:end],
                        goals_t[start:end],
                        self.height,
                        self.width,
                        radius,
                        limit,
                    )
                    windows[start:end] = current_windows
                    mask_codes[start:end] = current_codes

        with self.profiler.time("generate_observations/token_lookup"):
            idx = (windows.long() + self.coord_offset).clamp_(0, len(self.coord_lookup) - 1)
            cost2go_tokens = self.coord_lookup[idx]
            next_action_tokens = mask_codes.long() + self.next_action_base

        return cost2go_tokens, next_action_tokens


    def _get_agents_info(self, N, pos_t, goals_t, cost2go_tokens, next_action_tokens):
        limit = self.cfg.cost2go_value_limit
        agents_radius = self.cfg.agents_radius
        num_hist = self.cfg.num_previous_actions
        K = min(13, N)

        if getattr(self.cfg, "use_spatial_neighbors", False):
            from pogema_gpu.kernels.cuda_bfs import get_neighbors_spatial_cuda

            agents_indices = get_neighbors_spatial_cuda(
                pos_t, goals_t, self.action_history, next_action_tokens,
                self.coord_lookup, self.height, self.width, agents_radius,
                limit, self.coord_offset, self.pad_token, num_hist, K,
            )
        else:
            from pogema_gpu.kernels.cuda_bfs import get_neighbors_cuda

            agents_indices = get_neighbors_cuda(
                pos_t, goals_t, self.action_history, next_action_tokens,
                self.coord_lookup, agents_radius, limit, self.coord_offset,
                self.pad_token, self.inf_dist, num_hist, K,
            )

        obs = torch.cat([cost2go_tokens, agents_indices], dim=1)

        pad_len = self.cfg.context_size - obs.size(1)
        if pad_len > 0:
            final_pad = torch.full((N, pad_len), self.pad_token, dtype=torch.long, device=self.device)
            obs = torch.cat([obs, final_pad], dim=1)
        return obs
