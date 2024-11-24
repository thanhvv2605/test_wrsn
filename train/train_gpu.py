import sys
import os
import json
import torch

# Thêm thư mục cha vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Đường dẫn gốc
root_dir = os.getcwd()

from controller.DeepQ.DQNController import DQNController
from rl_env.WRSN import WRSN
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# Tải cấu hình từ tệp JSON
def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

# Tải cấu hình
config = load_config("params/deepq_models.json")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")
env = WRSN(
    scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml",
    agent_type_path="physical_env/mc/mc_types/default.yaml",
    num_agent=1,
    device=device
)
# Lấy trạng thái mẫu để xác định state_dim
state = env.reset()
if state["state"] is not None:
    sample_state = state["state"][0]
else:
    sample_state = env.get_state(agent_id=0)[0]
state_dim = sample_state.size
action_dim = len(env.net.listChargingLocations) + 1

controller = DQNController(
    num_agents=1,
    state_dim=state_dim,
    action_dim=action_dim,
    config=config,
    device=device
)
num_episodes = 1000
for episode in range(num_episodes):
    print(f"Episode {episode}")
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
            state = next_state
        else:
            next_state = env.step(None)
            done = next_state["terminal"]
            state = next_state

print("Training completed!")
# Lưu mô hình cho tất cả các agent
for agent_id in range(controller.num_agents):
    controller.save_model(agent_id, f"save_models/agent_{agent_id}_model.pth")