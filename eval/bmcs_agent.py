from open_spiel.python import rl_agent
from open_spiel.python.rl_environment import TimeStep

from .bmcs import BMCS


class BMCSAgent(rl_agent.AbstractAgent):
    """Agent that acts according to a BMCS"""

    def __init__(
        self,
        player_id: int,
        num_actions: int,
        bmcs: BMCS,
        *,
        use_ground_truth: bool = False,
    ):
        """Initialize the policy agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._bmcs = bmcs
        self._bmcs.belief_net.eval()
        self._bmcs.policy_net.eval()
        self._use_gt = use_ground_truth

    def step(self, time_step: TimeStep, is_evaluation=False):
        """Returns the action to be taken and updates the Q-values if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call. Ignored since this won't learn.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action. NOTE: always outputs full (9) action probs!
        """
        if time_step.last():
            return rl_agent.StepOutput(action=None, probs=None)

        action = self._bmcs.search(time_step, use_ground_truth=self._use_gt)

        # NOTE: should probably return a probability but BMCS search
        # doesn't. Tbf I'm not sure when we actually will need these probabilities lol.
        return rl_agent.StepOutput(action=action, probs=None)
