"""The corridor conflict: one grid, two expert solutions, one ambiguous state.

    row 0:  # # . # #      the passing bay at (0, 2)
    row 1:  . . . . .      the corridor
    row 2:  # # # # #

Agent A runs (1, 0) -> (1, 4) and agent B runs (1, 4) -> (1, 0), so one of them
must step into the bay. Both expert trajectories below solve that, and they
share exactly one state, at t=1, where either agent may yield. Actions follow
the POGEMA convention: 0 wait, 1 up, 2 down, 3 left, 4 right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WAIT, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
ACTION_DELTA = {WAIT: (0, 0), UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

RAW_GRID = ("##.##",
            ".....",
            "#####")
GRID = [[1 if cell == "#" else 0 for cell in row] for row in RAW_GRID]

A_START, A_GOAL = (1, 0), (1, 4)
B_START, B_GOAL = (1, 4), (1, 0)


@dataclass
class Trajectory:
    name: str
    actions: list            # actions[t] = (action_A, action_B) applied to the state at t
    positions: list = field(default=None)  # positions[t] = ((rowA, colA), (rowB, colB))

    def __post_init__(self):
        positions = [(A_START, B_START)]
        for a_action, b_action in self.actions:
            (a_row, a_col), (b_row, b_col) = positions[-1]
            a_delta, b_delta = ACTION_DELTA[a_action], ACTION_DELTA[b_action]
            positions.append(((a_row + a_delta[0], a_col + a_delta[1]),
                              (b_row + b_delta[0], b_col + b_delta[1])))
        self.positions = positions
        self._validate()

    def _validate(self):
        for step, (a_pos, b_pos) in enumerate(self.positions):
            if a_pos == b_pos:
                raise ValueError(f"{self.name}: vertex collision at t={step}")
            for row, col in (a_pos, b_pos):
                if GRID[row][col]:
                    raise ValueError(f"{self.name}: agent inside an obstacle at t={step}")
        for step in range(len(self.positions) - 1):
            before, after = self.positions[step], self.positions[step + 1]
            if before[0] == after[1] and before[1] == after[0]:
                raise ValueError(f"{self.name}: swap collision at t={step}")

    @property
    def horizon(self) -> int:
        return len(self.actions)


A_YIELDS = Trajectory("A_yields", [
    (RIGHT, LEFT),   # A (1,0)->(1,1), B (1,4)->(1,3)
    (RIGHT, WAIT),   # the ambiguous state resolves in A's favour
    (UP, LEFT),
    (WAIT, LEFT),
    (DOWN, LEFT),
    (RIGHT, WAIT),
    (RIGHT, WAIT),
])

B_YIELDS = Trajectory("B_yields", [
    (RIGHT, LEFT),
    (WAIT, LEFT),    # the ambiguous state resolves in B's favour
    (RIGHT, UP),
    (RIGHT, WAIT),
    (RIGHT, DOWN),
    (WAIT, LEFT),
    (WAIT, LEFT),
])

TRAJECTORIES = (A_YIELDS, B_YIELDS)

AMBIGUOUS_STEP = 1
AMBIGUOUS_STATE = ((1, 1), (1, 3))

# The four joint actions at the ambiguous state, in reporting order.
JOINT_ACTION_LABELS = {
    (WAIT, LEFT): "(wait, left)   [B_yields, valid]",
    (RIGHT, WAIT): "(right, wait)  [A_yields, valid]",
    (WAIT, WAIT): "(wait, wait)   [deadlock, invalid]",
    (RIGHT, LEFT): "(right, left)  [collision, invalid]",
}
VALID_JOINT_ACTIONS = ((WAIT, LEFT), (RIGHT, WAIT))

if A_YIELDS.positions[AMBIGUOUS_STEP] != AMBIGUOUS_STATE or \
        B_YIELDS.positions[AMBIGUOUS_STEP] != AMBIGUOUS_STATE:
    raise AssertionError("the two trajectories no longer share the ambiguous state")
