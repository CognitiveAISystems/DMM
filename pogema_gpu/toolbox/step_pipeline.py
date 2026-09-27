"""Observation -> proposal -> movement resolution -> shared commit."""
from ..movement import CSPIBTResolver


def validate_movement_backend(backend):
    if backend != "resolved":
        raise ValueError("evaluation requires resolved movement")


class StepPipeline:
    def __init__(self, batch, policy, *, movement_backend="resolved", validate=False):
        validate_movement_backend(movement_backend)
        self.batch, self.policy = batch, policy
        self.validate = validate
        self.resolver = (batch.native_resolver if policy.shield is None else
                         CSPIBTResolver(batch, policy.shield))

    def propose(self, inputs, ids):
        return self.policy.propose(inputs, ids)

    def resolve(self, output, ids):
        import torch
        proposals, scores = output
        b = self.batch
        slots = b.slots(ids)
        raw = torch.zeros_like(b.positions)
        raw[slots] = proposals
        preferences = None
        if self.policy.shield is not None:
            if scores is None or scores.shape != (*proposals.shape, 5):
                raise ValueError("movement resolver requires active-slot [B,N,5] scores")
            preferences = torch.zeros((*raw.shape, 5), device=b.device, dtype=torch.float32)
            preferences[slots] = scores
        move = self.resolver.resolve(raw, preferences, ids)
        # Track submitted actions after shielding, raw proposals otherwise.
        submitted = move.submitted
        self.policy.observation_builder.remember(
            submitted if self.policy.post_shield_history else raw, ids)
        return raw, submitted, move

    def commit(self, actions, move):
        self.batch.commit(move, validate=self.validate)
        self.resolver.after_commit()


def stage_seconds(events, *, shielded):
    """Report aggregate and non-overlapping stage timings."""
    stages = {key: events[i].elapsed_time(events[i+1])/1000 for i, key in enumerate(
        ("observation", "inference", "movement_resolution", "commit"))}
    stages["inference_and_shield"] = stages["inference"] + (stages["movement_resolution"] if shielded else 0)
    stages["simulation"] = stages["commit"] + (0 if shielded else stages["movement_resolution"])
    return stages
