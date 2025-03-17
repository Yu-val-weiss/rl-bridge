import numpy as np


class HandDealer:
    """Deals seeded hands to each player. If none, then returns a random choice."""

    _next_deal = None

    def seed_deal(self, deal: list[int]):
        """Seed the next deal to use. Should be in the deal action indices as used by tiny bridge."""
        if len(set(deal)) != 4:
            raise ValueError("must be 4 unique deals")
        self._next_deal = deal

    def __call__(self, state):
        """Sample a chance event in the given state."""
        l_a = state.legal_actions()

        # fallback if empty (needed for random reinitialisation as sometimes required)
        if self._next_deal is None or len(self._next_deal) == 0:
            actions, probs = zip(*state.chance_outcomes())
            return np.random.choice(actions, p=probs)

        # otherwise, work with the deal
        deal = self._next_deal[0]
        if deal not in l_a:
            raise ValueError(f"deal {deal} invalid out of legal deals {l_a}")

        self._next_deal = self._next_deal[1:]
        return deal
