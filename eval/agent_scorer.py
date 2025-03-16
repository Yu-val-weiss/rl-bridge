import copy
import logging
import random
from typing import Callable

import torch
from open_spiel.python.algorithms.random_agent import RandomAgent
from open_spiel.python.rl_agent import AbstractAgent
from open_spiel.python.rl_environment import Environment, TimeStep
from tqdm import trange

from models import PolicyNetwork

from .policy_agent import PolicyAgent

AgentFactory = Callable[[int], AbstractAgent]


class Scorer:
    def __init__(self, agent_a: AbstractAgent, agent_b: AbstractAgent) -> None:
        """Initialise the scorer

        Args:
            agent_a (AbstractAgent): agent to use for A, should be freshly initialised
            agent_b (AbstractAgen): agent to use for B, should be freshly initialised
        """
        self._agent_a = agent_a
        self._agent_b = agent_b

        self._agents = (self._agent_a, self._agent_b)

    def _shift_agents(self):
        """Shifts agents round 1 seat

        Args:
            clockwise (bool, optional): Clockwise means you increase the index.
        """
        fst, snd = self._agents
        self._agents = snd, fst

    def _play_step(self, time_step: TimeStep, player_id: int):
        agent = self._agents[player_id % 2]
        agent._player_id = player_id  # type: ignore # a little hacky but works, and means the agents are identical in each seat

        # is eval ignored by our agents, but used by e.g Q learner
        return agent.step(time_step, is_evaluation=True)

    def _play_game(self, env: Environment, time_step: TimeStep) -> list[float]:
        while not time_step.last():
            player_id: int = time_step.observations["current_player"]
            agent_output = self._play_step(time_step, player_id)
            if agent_output is None:
                logging.warning(
                    f"agent {player_id} ({type(player_id)}) returned None action"
                )
                action = random.choice(
                    time_step.observations["legal_actions"][player_id]
                )
            else:
                action = agent_output.action
            time_step = env.step([action])
        # step all at last time step, if they need to prepare for next episode
        for agent in self._agents:
            agent.step(time_step)

        return time_step.rewards

    def score(self, num_deals: int):
        env = Environment("tiny_bridge_4p")
        imps: list[tuple[int, int]] = []
        for _ in trange(num_deals):
            init_step = env.reset()
            init_state = copy.deepcopy(env.get_state)

            # do stuff
            rewards = self._play_game(env, init_step)

            # shift agents
            self._shift_agents()
            # go back to same deal, use init_step unchanged
            env.reset()
            env.set_state(init_state)
            second_rewards = self._play_game(env, init_step)

            # shift agents back
            self._shift_agents()

            # calculate imps, indices 0&2 and 1&3 are always the same score, so can use 0
            imp = calc_imp(rewards[0], second_rewards[0])
            imps.append(imp)

        a_imps, b_imps = zip(*imps)
        return (sum(a_imps) - sum(b_imps)) / num_deals


IMP_LOOKUP = [
    ((0, 19), 0),
    ((20, 40), 1),
    ((50, 80), 2),
    ((90, 120), 3),
    ((130, 160), 4),
    ((170, 210), 5),
    ((220, 260), 6),
    ((270, 310), 7),
    ((320, 360), 8),
    ((370, 420), 9),
    ((430, 490), 10),
    ((500, 590), 11),
    ((600, 740), 12),
    ((750, 890), 13),
    ((900, 1090), 14),
    ((1100, 1290), 15),
    ((1300, 1490), 16),
    ((1500, 1740), 17),
    ((1750, 1990), 18),
    ((2000, 2240), 19),
    ((2250, 2490), 20),
    ((2500, 2990), 21),
    ((3000, 3490), 22),
    ((3500, 3990), 23),
    ((4000, float("inf")), 24),
]


def calc_imp(a: float, b: float) -> tuple[int, int]:
    """Calculate IMP based on scores for the same hand (in the same seats)"""
    diff = abs(a - b)
    imp = 0
    for range_tuple, result in IMP_LOOKUP:
        if range_tuple[0] <= diff <= range_tuple[1]:
            imp = result

    return (0, imp) if a <= b else (imp, 0)


if __name__ == "__main__":

    def ragent():
        return RandomAgent(player_id=0, num_actions=9)

    def prlagent():
        net = PolicyNetwork(9, 84, 2048)
        d = torch.load("checkpoints/50k_test/step_50000/policy_net.pt")
        net.load_state_dict(d["state_dict"])
        return PolicyAgent(player_id=0, num_actions=9, policy_network=net)

    def pslagent():
        net = PolicyNetwork(9, 84, 2048)
        net.load_state_dict(torch.load("sl/policy_net.pt"))
        return PolicyAgent(player_id=0, num_actions=9, policy_network=net)

    num_runs = 10_000  # matches paper

    s = Scorer(prlagent(), ragent())
    print(f"A: RL, B: RAND = {s.score(num_runs)}")

    s = Scorer(pslagent(), ragent())
    print(f"A: SL, B: RAND = {s.score(num_runs)}")

    s = Scorer(pslagent(), prlagent())
    print(f"A: SL, B: RL = {s.score(num_runs)}")
