import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.getcwd()
import json

from controller.DeepQ.DQNController import DQNController
from rl_env.WRSN import WRSN

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
    sample_state = state["state"][0]
else:
    sample_state = env.get_state(agent_id=0)[0]
state_dim = sample_state.size
action_dim = len(env.net.listChargingLocations)+1
controller = DQNController(num_agents=1, state_dim=state_dim, action_dim=action_dim, config=config)

num_episodes = 5
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        agent_id = state["agent_id"]
        if agent_id is not None:
            prev_state = state["prev_state"]
            action = controller.select_action(agent_id, prev_state[0])
            next_state = env.step(action)
            reward = next_state["reward"]
            done = next_state["terminal"]
            if next_state["state"] is not None:
                next_state_flat = next_state["state"][0]
                controller.store_transition(agent_id, prev_state[0], action, reward, next_state_flat, done)
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