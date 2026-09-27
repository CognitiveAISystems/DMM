"""Optional policy-independent shield configuration and adapter boundary.

No model, Torch or CUDA is imported until tensor execution is requested.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShieldConfig:
    resolver: str = "none"
    repeat_escape: bool = False
    max_repeat_retries: int = 16
    rse_budget_bytes: int = 512*1024**2
    native_action_ties: bool = False

    def __post_init__(self):
        if self.resolver not in {"none", "sequential", "components"}:
            raise ValueError("PIBT must be none, sequential or components")
        if type(self.native_action_ties) is not bool:
            raise ValueError("native_action_ties must be boolean")
        if type(self.repeat_escape) is not bool:
            raise ValueError("repeat escape must be boolean")
        if self.repeat_escape and self.resolver == "none":
            raise ValueError("repeat escape requires PIBT")
        if type(self.max_repeat_retries) is not int or self.max_repeat_retries < 0:
            raise ValueError("RSE retries must be a nonnegative integer")
        if type(self.rse_budget_bytes) is not int or self.rse_budget_bytes < 1:
            raise ValueError("RSE history budget must be positive")

    def build(self, batch, *, cache=None):
        if self.resolver == "none":
            return None
        from .pibt import CUDAPIBT
        return CUDAPIBT(batch, resolver=self.resolver, cache=cache, profile=True,
                        repeat_escape=self.repeat_escape, max_repeat_retries=self.max_repeat_retries,
                        rse_budget_bytes=self.rse_budget_bytes,
                        native_action_ties=self.native_action_ties)

    def annotate(self, metadata, *, scores, history):
        if self.resolver != "none":
            metadata.update(history=history, pibt={"profile":"cuda-policy-shield-rse-v2",
                "resolver":self.resolver,"repeat_escape":self.repeat_escape,
                "max_repeat_retries":self.max_repeat_retries,"rse_budget_bytes":self.rse_budget_bytes,
                "priorities":"exact-bfs-fp32","ties":"stable-action-index-and-agent-id",
                "scores":scores,"repeat_identity":"exact-ordered-positions-fingerprint-prefilter",
                "retry_fallback":"prefer-highest-priority-unfinished-agent-moving-else-initial"})
            if self.native_action_ties:
                metadata["pibt"].update(ties="native-mt19937-uniform-float32-action-ties-v1",
                    rse_control=("cuda-device-loop-v1" if self.repeat_escape and self.resolver=="sequential" else "python-reference"),
                    priorities="exact-bfs-distance/10000-fp32;unreachable=free-cells",
                    agent_priority_ties="gcc11-std-sort-fresh-agent-index-order-v1",
                    tie_draw_order="agent-id,left,right,down,up,wait;walls-skipped",
                    tie_rng_lifecycle="episode-seed;redraw-every-resolution-including-RSE")
