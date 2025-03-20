# TinyBridge Bidding with Belief: a Systematic Evaluation of Belief Monte Carlo Search

This is the code for the R171 Reinforcement Learning mini project "TinyBridge Bidding with Belief: a Systematic Evaluation of Belief Monte Carlo Search". 
Belief Monte Carlo Search was introduced by Qiu et al. [1] to improve performance in the bidding phase of the partially observable game contract bridge.
Contract bridge stands out as an interesting game benchmark in RL due to its partial observability and the need for cooperation between partners.

This mini project systematically evaluates the system proposed in [1] against a suite of competing agents using common RL algorithms (Q-learning and Monte Carlo Tree Search).
OpenSpiel’s TinyBridge is used as the game environment [2]. We reimplemented the entire Belief Monte Carlo Search (BMCS) system from scratch, including database synthesis, a distributed Advantage Actor-Critic learning setup, and three distinct deep-RL networks.

## Installation

The environment with all necessary dependencies can be created using Poetry. Please follow this Poetry installation guide before resuming here: https://python-poetry.org/docs/.
After successfully installing poetry, run:

```bash
git clone https://github.com/Yu-val-weiss/rl-bridge.git
cd ./rl-bridge

poetry install
eval $(poetry env activate)
```

## Recreating our experiments

The training pipeline is the following:

![rl_new_pipe](https://github.com/user-attachments/assets/feb32395-4286-4a57-826b-653ed92da75b)

1. The `sl/` directory contains all code to learn the SL policy network:
   1. Generate synthetic dataset based on Q-Learner self-play using `sl/gen_dataset.py` or use our existing `sl/dataset.pt`. 
   2. Train the supervised learning (SL) policy network using `sl/train.py` or use our existing `sl/policy_net.pt`.
2. The `belief/` directory contains all code to learn the belief network from the SL policy network:
   1. Generate synthetic dataset based on SL policy network self-play using `belief/gen_dataset.py` or use our existing `belief/dataset.pt`. 
   2. Train the belief network using `belief/train.py` or use our existing `belief/belief_net_lr_0.001_dropout_0.1_hidden_20.pth`.
3. The `train.py` script starts a multi-thread Advantage Actor-Critic training using the `training/learner.py` and `training/actor.py`. 
   The hyperparameter configurations used are located in `configs/test.yaml` and can be modified.
   We have checkpointed our model at `checkpoints/50k_test/` after 50,000 training steps. 
5. The final agent evaluation can be run in `eval/agent_scorer.py`. 

## References
[1] Z. Qiu, S. Wang, D. You, and M. Zhou, ‘Bridge
Bidding via Deep Reinforcement Learning
and Belief Monte Carlo Search,’ IEEE/CAA
Journal of Automatica Sinica, vol. 11, no. 10,
pp. 2111–2122, Oct. 2024, issn: 2329-9274.
doi: 10.1109/JAS.2024.124488. [Online].
Available: https://ieeexplore.ieee.org/
document/10664606/?arnumber=10664606.

[2] M. Lanctot, E. Lockhart, J.-B. Lespiau, et al.,
‘OpenSpiel: A framework for reinforcement
learning in games,’ CoRR, vol. abs/1908.09453,
2019. arXiv: 1908.09453 [cs.LG]. [Online].
Available: http://arxiv.org/abs/1908.
09453.
