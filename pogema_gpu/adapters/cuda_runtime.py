"""Shared episode lifecycle for CUDA policies, independent of neural inference."""

from ..observations.cuda import CUDADMMTokenizer, validate_bfs_options
from ..planning.policy_shield import ShieldConfig


class CUDAPolicyRuntime:
    """Own tokenizer, episode RNG resets and optional shield for one batch.

    The factory creates an episode's RNG state without drawing actions. The
    field name preserves the adapters' public samplers/generators attributes.
    Checkpoint loading, inference and action-history semantics stay in adapters.
    """

    always_post_shield_history = False

    @property
    def post_shield_history(self):
        return self.always_post_shield_history or self.shield_config.repeat_escape

    def __init__(self, *, rng_factory, rng_attribute="samplers", pibt="none",
                 repeat_escape=False, max_repeat_retries=16,
                 rse_budget_bytes=512*1024**2, profile_bfs=False,
                 bfs_chunk_size=128, bfs_budget_bytes=256*1024**2,
                 native_action_ties=False):
        self.shield_config = ShieldConfig(pibt, repeat_escape, max_repeat_retries, rse_budget_bytes,
                                          native_action_ties)
        validate_bfs_options(bfs_chunk_size, bfs_budget_bytes)
        self._rng_factory = rng_factory
        self._rng_attribute = rng_attribute
        self.pibt = pibt
        self.profile_bfs = profile_bfs
        self.bfs_chunk_size = bfs_chunk_size
        self.bfs_budget_bytes = bfs_budget_bytes
        self.shield = None
        self.inference_profiler = None
        self.bfs_cache_options = {}

    def configure_cost_to_go(self, *, cache_mode="bounded"):
        """Select optional map analysis before reset, without changing policy/RNG."""
        from ..observations.services import MapServices
        if getattr(self,"batch",None) is not None:
            raise ValueError("configure cost-to-go before attaching an environment batch")
        MapServices(None, cache_mode=cache_mode)
        self.bfs_cache_options = dict(cache_mode=cache_mode)

    def reset(self, batch):
        if batch.device != self.device:
            raise ValueError("Policy and CUDA batch must use the same device")
        # A new batch replaces the previous service, not a second simultaneous
        # distance cache. Release our owners before measuring admission headroom.
        self.shield = None
        self.tokenizer = self.observation_builder = self.map_services = None
        self.batch = batch
        options = dict(profile=self.profile_bfs,chunk_size=self.bfs_chunk_size,budget_bytes=self.bfs_budget_bytes)
        if self.bfs_cache_options:
            from ..observations.services import MapServices
            self.tokenizer = CUDADMMTokenizer(batch,services=MapServices(batch,**options,**self.bfs_cache_options))
        else:
            self.tokenizer = CUDADMMTokenizer(batch,**options)
        self.observation_builder = self.tokenizer
        self.map_services = self.observation_builder.services
        states = [self._rng_factory(task.policy_seed, self.device) for task in batch.tasks]
        setattr(self, self._rng_attribute, states)
        self.generations = batch.generations.copy()
        self.shield = self.shield_config.build(batch, cache=self.tokenizer.cache)

    def prepare(self, ids):
        states = getattr(self, self._rng_attribute)
        for i in ids:
            if self.generations[i] != self.batch.generations[i]:
                states[i] = self._rng_factory(self.batch.tasks[i].policy_seed, self.device)
                self.generations[i] = self.batch.generations[i]
        return self.tokenizer.prepare(ids)
