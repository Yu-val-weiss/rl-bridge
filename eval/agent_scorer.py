import copy
import logging
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pyspiel
import torch
from open_spiel.python.algorithms.mcts import MCTSBot, RandomRolloutEvaluator
from open_spiel.python.algorithms.mcts_agent import MCTSAgent
from open_spiel.python.algorithms.random_agent import RandomAgent
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.rl_agent import AbstractAgent
from open_spiel.python.rl_environment import Environment, TimeStep
from scipy.stats import sem, t
from tqdm import tqdm, trange

from models import BeliefNetwork, PolicyNetwork

from .bmcs import BMCS
from .bmcs_agent import BMCSAgent
from .policy_agent import PolicyAgent

AgentFactory = Callable[[int], AbstractAgent]


class EvalScore(NamedTuple):
    mean: float
    stdev: float
    conf_interval: tuple[float, float]
    conf_band: float

    def __str__(self) -> str:
        return f"μ: {self.mean:6f} ± {self.conf_band:.6f}, σ: {self.stdev:.6f}, [{self.conf_interval[0]:.6f}, {self.conf_interval[1]:.6f}]"


class Scorer:
    def __init__(
        self,
        agent_a: AbstractAgent,
        agent_b: AbstractAgent,
        *,
        include_full_state=False,
    ) -> None:
        """Initialise the scorer

        Args:
            agent_a (AbstractAgent): agent to use for A, should be freshly initialised
            agent_b (AbstractAgen): agent to use for B, should be freshly initialised
        """
        self._agent_a = agent_a
        self._agent_b = agent_b

        self._agents = (self._agent_a, self._agent_b)

        self.env = Environment("tiny_bridge_4p", include_full_state=include_full_state)

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

    def score(self, num_deals: int) -> tuple[EvalScore, np.ndarray]:
        imps = np.zeros((num_deals, 2), dtype=int)
        for i in trange(num_deals, desc="playing games...", leave=False):
            init_step = self.env.reset()
            init_state = copy.deepcopy(self.env.get_state)

            # do stuff
            rewards = self._play_game(self.env, init_step)

            # shift agents
            self._shift_agents()
            # go back to same deal, use init_step unchanged
            self.env.reset()
            self.env.set_state(init_state)
            second_rewards = self._play_game(self.env, init_step)

            # shift agents back
            self._shift_agents()

            # calculate imps, indices 0&2 and 1&3 are always the same score, so can use 0
            imps[i] = calc_imp(rewards[0], second_rewards[0])

        a_imps, b_imps = imps[:, 0], imps[:, 1]
        imp_diff = a_imps - b_imps
        mean = np.mean(imp_diff)
        std = np.std(imp_diff, ddof=1)
        interval = t.interval(0.95, df=num_deals - 1, loc=mean, scale=sem(imp_diff))
        return EvalScore(
            mean=float(mean),
            stdev=float(std),
            conf_interval=interval,
            conf_band=abs(interval[1] - mean),
        ), imps


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
    NUM_ROLLOUTS = 30

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

    def bmcsagent(*, use_ground_truth=False):
        belief_net = BeliefNetwork(hidden_size=20)
        belief_net.load_state_dict(
            torch.load("belief/belief_net_lr_0.001_dropout_0.1_hidden_20.pth")
        )
        policy_net = PolicyNetwork(9, 84, 2048)
        policy_net.load_state_dict(torch.load("sl/policy_net.pt"))
        bmcs = BMCS(
            belief_net, policy_net, [], r_max=NUM_ROLLOUTS, p_max=NUM_ROLLOUTS, k=9
        )
        return BMCSAgent(0, 9, bmcs, use_ground_truth=use_ground_truth)

    def qagent_warmup(warmup: int = 1_000):
        q = QLearner(0, num_actions=9)
        r = RandomAgent(0, num_actions=9)
        agents = [q, r]
        env = Environment("tiny_bridge_4p")
        for _ in trange(warmup, desc="warming up q learner"):
            t_s = env.reset()
            while not t_s.last():
                player_id = t_s.current_player()
                agent = agents[player_id % 2]
                agent._player_id = player_id
                agent_output = agent.step(t_s)
                if agent_output is None:
                    raise ValueError("got unexpected None output")
                t_s = env.step([agent_output.action])
            for a in agents:
                a.step(t_s)
        return q

    QAGENT_WU_STEPS = 10_000
    q = qagent_warmup(QAGENT_WU_STEPS)

    def qagent():
        return q

    def mcts():
        bot = MCTSBot(
            pyspiel.load_game("tiny_bridge_4p"),
            uct_c=math.sqrt(2),
            max_simulations=NUM_ROLLOUTS,
            evaluator=RandomRolloutEvaluator(NUM_ROLLOUTS),
        )
        return MCTSAgent(0, 9, bot)

    num_runs = 1_000

    results = [f"EVALUATION: {num_runs} games"]

    agents_to_eval = [
        (ragent, "RAND"),
        (pslagent, "SL"),
        (qagent, f"Q({QAGENT_WU_STEPS})"),
        (mcts, "MCTS"),
        (lambda: bmcsagent(use_ground_truth=False), "BMCS"),
        (lambda: bmcsagent(use_ground_truth=True), "BMCS(gt)"),
    ]

    n = len(agents_to_eval)
    total = (n * (n - 1)) // 2

    pbar = tqdm(
        combinations(agents_to_eval, 2), total=total, desc="Evaluating agents..."
    )

    imp_results = {}
    imp_dest = Path("eval/imp_arrs.npz")

    for (agent_a, a_name), (agent_b, b_name) in pbar:
        s = Scorer(agent_a(), agent_b(), include_full_state=True)
        score, imp_arr = s.score(num_runs)
        imp_results[f"{a_name.lower()}:{b_name.lower()}"] = imp_arr
        score_str = f"A: {a_name} vs B: {b_name} = {score}"
        results.append(score_str)
        pbar.write(score_str)

        with open("eval/results_w_extra_info.txt", "w") as f:
            f.write("\n".join(results))

        np.savez_compressed(imp_dest, **imp_results)
