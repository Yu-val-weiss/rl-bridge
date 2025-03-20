import pyperf
import torch
from open_spiel.python.rl_environment import Environment

from eval.bmcs import BMCS
from models import BeliefNetwork, PolicyNetwork

if __name__ == "__main__":
    belief_net = BeliefNetwork(hidden_size=20)
    belief_net.load_state_dict(
        torch.load("belief/belief_net_lr_0.001_dropout_0.1_hidden_20.pth")
    )
    policy_net = PolicyNetwork(9, 84, 2048)
    policy_net.load_state_dict(torch.load("sl/policy_net.pt"))

    env = Environment("tiny_bridge_4p")

    runner = pyperf.Runner()
    for i in range(10, 110, 10):
        ts = env.reset()
        bmcs = BMCS(belief_net, policy_net, [], r_max=i, p_max=i, k=9)
        runner.timeit(f"BMCS {i}", stmt="bmcs.search(ts)", globals=locals())
