import torch
from open_spiel.python import rl_agent
from open_spiel.python.rl_environment import TimeStep

from models import PolicyNetwork
from utils import get_device, mask_action_probs, random_argmax


class PolicyAgent(rl_agent.AbstractAgent):
    """Agent that acts from a simple policy network"""

    def __init__(
        self,
        player_id: int,
        num_actions: int,
        policy_network: PolicyNetwork,
    ):
        """Initialize the policy agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._net = policy_network
        self._net.eval()

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

        info_state = time_step.observations["info_state"][self._player_id]
        obs = torch.tensor(info_state, dtype=torch.float32, device=get_device())
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        probs = self._net(obs).cpu().detach().numpy()
        legal_action_probs = mask_action_probs(probs, legal_actions)
        action = random_argmax(legal_action_probs)  # greedy algorithm

        return rl_agent.StepOutput(action=action, probs=legal_action_probs)
