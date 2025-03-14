import pyspiel
import torch
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from tqdm import trange

# Load the game
game = pyspiel.load_game("tiny_bridge_4p")

# Set up the environment
env = rl_environment.Environment(game)

# Initialize the Q-learning agents
agents = [
    tabular_qlearner.QLearner(player_id=i, num_actions=env.action_spec()["num_actions"])
    for i in range(4)
]

# Training loop
num_episodes = 20_000
store_from = 10_000
info_states = []
actions = []

# Collect data for PyTorch dataset
for episode in (pb := trange(num_episodes)):
    trajectory = []
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        if agent_output is None:
            continue

        action = agent_output.action
        time_step = env.step([agent_output.action])

        if episode >= store_from:
            # Assuming info_state is a vector; convert to numpy array (modify if different structure)
            info_state = time_step.observations["info_state"][player_id]
            info_states.append(torch.tensor(info_state))  # Convert to float32 tensor
            actions.append(action)  # Keep action as an integer (long)

    # Step all agents
    for a in agents:
        a.step(time_step)

# Convert lists to PyTorch tensors after collection
info_states_tensor = torch.stack(info_states)
actions_tensor = torch.tensor(actions, dtype=torch.long)

# Save the tensors to a file
torch.save(
    {"info_states": info_states_tensor, "actions": actions_tensor},
    "sl/dataset.pt",
)

print("Dataset has been saved as 'sl/dataset.pt'")
