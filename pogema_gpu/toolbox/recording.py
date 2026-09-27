"""Record CUDA episodes for terminal result export."""

import torch

from ..simulator import EXPORT_FIELDS, export_transitions
from ..tasks import content_hash


class EpisodeRecorder:
    def __init__(self, tasks, batch, *, shielded, record_trace):
        self.tasks, self.batch = tasks, batch
        shape = (len(tasks), max(t.horizon for t in tasks), batch.n)
        self.actions = torch.empty(shape, dtype=torch.uint8, device=batch.device)
        self.proposals = torch.empty_like(self.actions) if shielded else self.actions
        self.terminal = {name: getattr(batch, name).new_empty((len(tasks), *getattr(batch, name).shape[1:]))
                         for name in EXPORT_FIELDS}
        self.trace = ({name: getattr(batch, name).new_empty(shape[:2] + getattr(batch, name).shape[1:])
                       for name in ("positions", "executed", "solved")} if record_trace else {})

    def step(self, rows, slots, steps, raw, actions):
        self.actions[rows, steps] = actions[slots].to(torch.uint8)
        if self.proposals is not self.actions:
            self.proposals[rows, steps] = raw[slots].to(torch.uint8)
        for name, buffer in self.trace.items():
            buffer[rows, steps] = getattr(self.batch, name)[slots]

    def finish(self, rows, slots):
        for name, buffer in self.terminal.items():
            buffer[rows] = getattr(self.batch, name)[slots]

    def export(self, metadata):
        state = {name: tensor.cpu().tolist() for name, tensor in self.terminal.items()}
        terminal = export_transitions(self.tasks, self.batch.width, self.batch.radius, state)
        # Trim on CPU tensors before converting to Python; never read unwritten padding.
        actions = self.actions.cpu()
        proposals = self.proposals.cpu() if self.proposals is not self.actions else actions
        traces = {name: buffer.cpu() for name, buffer in self.trace.items()}
        records = []
        for i, task in enumerate(self.tasks):
            length = state["steps"][i]
            tape = actions[i, :length].tolist()
            record = {"task_id":task.task_id, "layout_hash":task.layout_hash, "task":task.to_dict(),
                      "profile":task.profile, "horizon":task.horizon, "status":"completed",
                      "policy_seed":task.policy_seed, "policy":metadata,
                      "env_grid_search":{"map_name":task.task_id, "num_agents":task.num_agents},
                      "metrics":terminal[i]["metrics"], "steps":length, "actions_hash":content_hash(tape),
                      "actions":tape, "final_positions":terminal[i]["positions"]}
            if proposals is not actions:
                raw = proposals[i, :length].tolist()
                record["policy_proposals"] = raw
                record["shield_overrides"] = sum(a != p for row, prop in zip(tape, raw) for a, p in zip(row, prop))
            if traces:
                rows = {name: buffer[i, :length].tolist() for name, buffer in traces.items()}
                record["trace"] = [
                    {"positions":[[p//self.batch.width-self.batch.radius, p%self.batch.width-self.batch.radius]
                                  for p in rows["positions"][s]],
                     "executed_actions":rows["executed"][s],
                     "rewards":[float(rows["solved"][s])]*task.num_agents,
                     "terminated":[rows["solved"][s]]*task.num_agents,
                     "truncated":[s+1 >= task.horizon]*task.num_agents,
                     "metrics":terminal[i]["metrics"] if s == length-1 else None} for s in range(length)]
            records.append(record)
        return records
