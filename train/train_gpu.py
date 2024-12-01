import sys
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import math

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

num_episodes = 100
writer = SummaryWriter(log_dir="logs")  # Khởi tạo TensorBoard writer

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0  # Tổng phần thưởng trong tập

    while not done:
        agent_id = state["agent_id"]
        if agent_id is not None:
            prev_state_data = state["prev_state"][0]
            prev_state = torch.tensor(prev_state_data, dtype=torch.float32, device=device)

            # Lựa chọn hành động từ bộ điều khiển
            action = controller.select_action(agent_id, prev_state)
            # Gọi hàm step với agent_id và action
            next_state = env.step(agent_id, action)
            reward = next_state["reward"]
            done = next_state["terminal"]

            if next_state["state"] is not None:
                next_state_data = next_state["state"][0]
                next_state_flat = torch.tensor(next_state_data, dtype=torch.float32, device=device)

                controller.store_transition(agent_id, prev_state, action, reward, next_state_flat, done)
                controller.train_agent(agent_id)
                controller.sync_target_network(agent_id)
            state = next_state
            if reward is not None:
                total_reward += reward  # Cộng phần thưởng
        else:
            # Khi agent_id là None, chúng ta cần tiếp tục bước môi trường mà không thực hiện hành động
            # Gọi hàm step với agent_id là None và action là None
            next_state = env.step(None, None)
            done = next_state["terminal"]
            state = next_state

    # Ghi phần thưởng và epsilon vào TensorBoard
    writer.add_scalar('Total Reward/Episode', total_reward, episode)
    eps_threshold = controller.epsilon_end + (controller.epsilon_start - controller.epsilon_end) * \
                    math.exp(-1. * controller.episode_count / controller.epsilon_decay)
    writer.add_scalar('Epsilon/Episode', eps_threshold, episode)

    # Sau mỗi tập, tăng số tập trong controller
    controller.increment_episode()
    print(f"Episode {episode+1}/{num_episodes} completed! Epsilon: {eps_threshold}")

writer.close()

print("Training completed!")
for agent_id in range(controller.num_agents):
    controller.save_model(agent_id, f"save_models/agent_{agent_id}_model.pth")