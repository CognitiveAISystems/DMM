"""Exact, episode-local repeat avoidance around a one-step PIBT resolver.

No network calls or simulator transitions occur during retries. Fingerprints
only select history candidates; full ordered positions decide equality.

"""
import torch


def validate_rse(enabled=False, max_retries=16, budget_bytes=512*1024**2):
    if type(enabled) is not bool:
        raise ValueError("repeat escape must be boolean")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("RSE retries must be a nonnegative integer")
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError("RSE history budget must be positive")


class RepeatStateEscape:
    COUNTERS = ("repeat_detections", "repeat_retries", "repeat_escapes", "repeat_unescaped")

    def __init__(self, positions, horizon, *, max_retries=16, budget_bytes=512*1024**2):
        validate_rse(True, max_retries, budget_bytes)
        if type(horizon) is not int or horizon < 1:
            raise ValueError("RSE horizon must be positive")
        if positions.ndim != 2 or positions.dtype != torch.long or min(positions.shape) < 1:
            raise ValueError("RSE positions must be nonempty int64 [B,N]")
        torch._assert_async(((positions >= 0) & (positions < 2**31)).all(), "RSE cell IDs must fit int32")
        b, n = positions.shape
        required = b*(horizon+1)*(n*4+8)
        if required > budget_bytes:
            raise ValueError(f"RSE exact history requires {required} bytes; budget is {budget_bytes}")
        self.max_retries = max_retries
        self.history = torch.zeros((b, horizon+1, n), dtype=torch.int32, device=positions.device)
        self.hashes = torch.zeros((b, horizon+1), dtype=torch.int64, device=positions.device)
        self.lengths = torch.zeros(b, dtype=torch.long, device=positions.device)
        self.rows = torch.arange(b, device=positions.device)
        self.times = torch.arange(horizon+1, device=positions.device)
        # Private CPU RNG: independent of policy seeds, batch order and global RNG.
        rng = torch.Generator().manual_seed(0x5EED5EED)
        self.coefficients = torch.randint(1, 2**62, (n,), generator=rng).to(positions.device)
        self.stats = torch.zeros((b, 4), dtype=torch.long, device=positions.device)
        self.pending_stats = torch.zeros_like(self.stats)
        self.reset_at(list(range(b)), positions)

    def fingerprint(self, positions):
        return (positions.long()*self.coefficients).sum(-1)

    def reset_at(self, ids, positions):
        self.history[ids, 0] = positions[ids].int()
        self.hashes[ids, 0] = self.fingerprint(positions[ids])
        self.lengths[ids] = 1
        self.stats[ids] = 0
        self.pending_stats[ids] = 0

    def repeated(self, positions, active):
        matches = ((self.hashes == self.fingerprint(positions)[:, None])
                   & (self.times[None] < self.lengths[:, None]) & active[:, None])
        # One scalar synchronization; full positions never move to the host.
        repeated = torch.zeros_like(active)
        if bool(matches.any().item()):
            env, time = matches.nonzero(as_tuple=True)
            exact = (self.history[env, time] == positions[env]).all(-1)
            counts = torch.zeros_like(self.lengths)
            counts.scatter_add_(0, env, exact.long())
            repeated = counts > 0
        return repeated

    def actions(self, positions, goals, valid, order, active, resolve):
        """resolve(forbidden) returns actions/next IDs without updating priorities."""
        forbidden = torch.zeros_like(valid)
        actions, next_ids = resolve(forbidden)
        repeated = self.repeated(next_ids, active)
        detected = repeated.clone()
        retries = torch.zeros_like(self.lengths)
        unfinished = positions != goals
        first = unfinished.gather(1, order).long().argmax(-1)
        escape_agent = order.gather(1, first[:, None]).squeeze(1)
        best_actions, best_ids = actions.clone(), next_ids.clone()
        best_moves = unfinished.any(-1) & (next_ids[self.rows, escape_agent] != positions[self.rows, escape_agent])
        for _ in range(self.max_retries):
            chosen_forbidden = forbidden.gather(2, actions[:, :, None]).squeeze(-1)
            available = (valid & ~forbidden).sum(-1)
            eligible = unfinished & ~chosen_forbidden & (available > 1)
            retry = repeated & eligible.any(-1)
            if not bool(retry.any().item()):
                break
            rank = eligible.gather(1, order).long().argmax(-1)
            agent = order.gather(1, rank[:, None]).squeeze(1)
            forbidden[self.rows, agent, actions[self.rows, agent]] |= retry
            candidate_actions, candidate_ids = resolve(forbidden)
            actions = torch.where(retry[:, None], candidate_actions, actions)
            next_ids = torch.where(retry[:, None], candidate_ids, next_ids)
            retries += retry.long()
            repeated = self.repeated(next_ids, detected)
            moved = next_ids[self.rows, escape_agent] != positions[self.rows, escape_agent]
            improve = repeated & moved & ~best_moves
            best_actions = torch.where(improve[:, None], actions, best_actions)
            best_ids = torch.where(improve[:, None], next_ids, best_ids)
            best_moves |= improve
        # Match the inspected reference's bounded-retry fallback: prefer a
        # candidate moving the highest-priority unfinished agent, else initial.
        actions = torch.where(repeated[:, None], best_actions, actions)
        next_ids = torch.where(repeated[:, None], best_ids, next_ids)
        self.pending_stats = torch.stack((detected.long(), retries,
                                         (detected & ~repeated).long(), repeated.long()), -1)
        return actions, next_ids

    def commit(self, positions, changed):
        torch._assert_async((~changed | (self.lengths < self.history.shape[1])).all(),
                            "RSE history horizon exhausted")
        time = self.lengths.clamp_max(self.history.shape[1]-1)
        self.history[self.rows, time] = torch.where(changed[:, None], positions.int(), self.history[self.rows, time])
        self.hashes[self.rows, time] = torch.where(changed, self.fingerprint(positions), self.hashes[self.rows, time])
        self.lengths += changed.long()
        self.stats += self.pending_stats*changed[:, None]
        self.pending_stats.zero_()

    def statistics(self, slot):
        return dict(zip(self.COUNTERS, self.stats[slot].cpu().tolist()))
