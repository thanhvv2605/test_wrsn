import sys
import os
import numpy as np
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.DeepQ.DQNController import DQNController
from rl_env.WRSN import WRSN

# Load configuration from JSON file
def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

# Load configuration
config = load_config("params/deepq_models.json")

# Initialize the WRSN environment
network = WRSN(
    scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml",
    agent_type_path="physical_env/mc/mc_types/default.yaml",
    num_agent=1
)

# Reset the environment and get initial state
state = network.reset()
if state["state"] is not None:
    sample_state_flat, embedding_dim = state["state"]
else:
    sample_state_flat, embedding_dim = network.get_state(agent_id=0)

state_dim = sample_state_flat.size  # Size of the numpy array

# Include the base station action in action_dim
action_dim = len(network.net.listChargingLocations) + 1  # +1 for the base station action

# Initialize the DQNController
controller = DQNController(num_agents=1, state_dim=state_dim, action_dim=action_dim, config=config)

# Load the saved model for agent 0
controller.load_model(agent_id=0, path="save_models/agent_0_model.pth")

# Reset the environment to start the evaluation
request = network.reset()

while not request["terminal"]:
    print("Agent ID:", request["agent_id"], "Action:", request["action"], "Terminal:", request["terminal"])
    if request["agent_id"] is not None:
        # Extract the flattened state for the agent
        state_flat, embedding_dim = request["state"]
        # Convert state_flat to torch tensor if necessary
        if not isinstance(state_flat, torch.Tensor):
            state_flat = torch.tensor(state_flat, dtype=torch.float32)
        # Select an action using the loaded model
        action = controller.select_action(request["agent_id"], state_flat)
        # Step the environment with the selected action
        request = network.step(request["agent_id"], action)
    else:
        # If no agent is ready to act, proceed without an action
        request = network.step(None, None)

# Print the total simulation time
print("Total simulation time:", network.net.env.now)