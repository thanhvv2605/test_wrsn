import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.getcwd()
import json

from controller.DeepQ.DQNController import DQNController
from rl_env.WRSN import WRSN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration from JSON file
def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

# Load configuration
config = load_config("params/deepq_models.json")

# Initialize environment
env = WRSN(
    scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml",
    agent_type_path="physical_env/mc/mc_types/default.yaml",
    num_agent=1
)

# Get a sample state to determine state_dim
state = env.reset()
if state["state"] is not None:
    sample_state = torch.tensor(state["state"][0], dtype=torch.float32).to(device)
else:
    sample_state = torch.tensor(env.get_state(agent_id=0)[0], dtype=torch.float32).to(device)

state_dim = sample_state.size(0)  # Use PyTorch tensor size
action_dim = len(env.net.listChargingLocations) + 1
controller = DQNController(num_agents=1, state_dim=state_dim, action_dim=action_dim, config=config)
controller.q_networks = [q_net.to(device) for q_net in controller.q_networks]
controller.target_networks = [target_net.to(device) for target_net in controller.target_networks]

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        agent_id = state["agent_id"]
        if agent_id is not None:
            prev_state = torch.tensor(state["prev_state"][0], dtype=torch.float32).to(device)
            action = controller.select_action(agent_id, prev_state)
            next_state = env.step(action)
            reward = next_state["reward"]
            done = next_state["terminal"]

            if next_state["state"] is not None:
                next_state_flat = torch.tensor(next_state["state"][0], dtype=torch.float32).to(device)
                controller.store_transition(agent_id, prev_state, action, reward, next_state_flat, done)
                controller.train_agent(agent_id)
                controller.sync_target_network(agent_id)
            else:
                # Agent is still moving or charging
                pass
            state = next_state
        else:
            next_state = env.step(None)
            done = next_state["terminal"]
            state = next_state

# Save models for all agents
for agent_id in range(controller.num_agents):
    controller.save_model(agent_id, f"save_models/agent_{agent_id}_model.pth")