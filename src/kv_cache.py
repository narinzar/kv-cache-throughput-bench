"""Bounded KV cache eviction policies and a hit-rate simulator.

A "hit rate" for a bounded cache is defined as the fraction of the full
(unbounded) attention probability mass that lands on positions still resident
in a cache of budget K. This is measured against the model's real attention
distributions, so it reflects how much attention each policy actually preserves.

Policies:
  none : keep everything (upper bound, hit rate = 1.0)
  fifo : when full, evict the oldest position index
  lru  : when full, evict the position whose last "access" is oldest, where a
         position is accessed on a step if it receives the largest share of that
         step's attention mass
"""
from collections import OrderedDict


class EvictionSimulator:
    """Replays generation steps and tracks which positions a size-K cache holds.

    step(attn_row) takes the full attention distribution over positions
    [0..t] for the current query and returns the hit rate for this step.
    """

    def __init__(self, budget, policy):
        assert policy in ("none", "fifo", "lru")
        self.budget = budget
        self.policy = policy
        self.resident = OrderedDict()  # position -> last_access_step
        self.t = 0

    def _evict_one(self):
        if self.policy == "fifo":
            # oldest inserted == smallest position index
            victim = next(iter(self.resident))
        else:  # lru
            victim = min(self.resident, key=lambda p: self.resident[p])
        del self.resident[victim]

    def step(self, attn_row):
        """attn_row: list/1D tensor of attention mass over positions 0..t.

        The new position t is inserted first (the model always keeps the current
        token), then eviction runs if over budget, then the hit rate is the mass
        on resident positions.
        """
        t = self.t
        self.resident[t] = t  # insert current position, accessed now
        # update last-access using this step's attention (argmax position)
        if len(attn_row) > 0:
            top = int(max(range(len(attn_row)), key=lambda i: attn_row[i]))
            if top in self.resident:
                self.resident[top] = t
        if self.policy != "none":
            while len(self.resident) > self.budget:
                self._evict_one()
        mass = sum(attn_row[p] for p in self.resident if p < len(attn_row))
        total = sum(attn_row) or 1.0
        self.t += 1
        return mass / total
