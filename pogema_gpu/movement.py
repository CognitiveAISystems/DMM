"""Alternative movement resolvers; neither owns observations or neural inference.

Importing this module does not import Torch or compile CUDA. A resolved move
belongs to one batch state and must be committed before resolving another step.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedMove:
    batch: object
    state_version: int
    positions: object
    actions: object
    submitted: object


class NativeSoftResolver:
    name = "native-soft"

    def __init__(self, batch):
        self.batch = batch
        self.scratch = self.positions = self.actions = None
        self._resolved_version = None

    def resolve(self, proposals, scores=None, ids=None):
        import torch
        b = self.batch
        if self._resolved_version == b.state_version:
            raise ValueError("commit or reset before resolving the same state again")
        b.validate_actions(proposals)
        if self.scratch is None:
            self.scratch = torch.empty((b.b, 3*b.cells+4*b.n), dtype=torch.int32, device=b.device)
            self.positions = torch.empty_like(b.positions)
            self.actions = torch.empty_like(b.executed)
        b.kernel.resolve_soft(b.walls, b.positions, proposals, self.actions,
                              self.positions, b.finished, self.scratch)
        self._resolved_version = b.state_version
        return ResolvedMove(b, b.state_version, self.positions, self.actions, proposals)

    def after_commit(self):
        pass


class CSPIBTResolver:
    name = "cs-pibt"

    def __init__(self, batch, shield):
        if shield.batch is not batch:
            raise ValueError("movement resolver and shield must share a batch")
        self.batch, self.shield = batch, shield

    def resolve(self, proposals, scores=None, ids=None):
        if scores is None:
            raise ValueError("CS-PIBT requires policy preference scores")
        self.batch.validate_actions(proposals)
        actions = self.shield.actions(scores, ids)
        # The submitted tape is the shield output. Policy overrides are NOT
        # native collision repairs; the evaluator records them separately.
        return ResolvedMove(self.batch, self.batch.state_version,
                            self.shield.pending_positions, actions, actions)

    def after_commit(self):
        self.shield.commit()


def validate_resolved_move(move):
    """Optional debug gate: reject, never repair, an invalid resolved transition.

    O(N log N) sorting uses agent-sized buffers, not map-sized scratch. Assertions
    stay on-device. Normal trusted resolver execution need not run this every step.
    """
    import torch
    b = move.batch
    b.validate_actions(move.actions)
    p, q = b.positions, move.positions
    active = ~b.finished[:, None]
    valid_cell = (q >= 0) & (q < b.cells)
    torch._assert_async((valid_cell | ~active).all(), "resolved cell out of bounds")
    safe = q.clamp(0, b.cells-1)
    dr, dc = safe//b.width-p//b.width, safe%b.width-p%b.width
    legal = ((move.actions == 0) & (dr == 0) & (dc == 0)
             | (move.actions == 1) & (dr == -1) & (dc == 0)
             | (move.actions == 2) & (dr == 1) & (dc == 0)
             | (move.actions == 3) & (dr == 0) & (dc == -1)
             | (move.actions == 4) & (dr == 0) & (dc == 1))
    torch._assert_async((legal | ~active).all(), "resolved action/position mismatch")
    torch._assert_async((~b.walls.flatten(1).gather(1, safe) | ~active).all(), "resolved wall collision")
    ordered = safe.sort(-1).values
    torch._assert_async(((ordered[:, 1:] != ordered[:, :-1]) | ~active).all(), "resolved vertex collision")
    old, ids = p.sort(-1)
    index = torch.searchsorted(old, safe).clamp_max(b.n-1)
    occupant = ids.gather(1, index)
    swap = (old.gather(1, index) == safe) & (safe != p) & (safe.gather(1, occupant) == p)
    torch._assert_async((~swap | ~active).all(), "resolved edge swap")
