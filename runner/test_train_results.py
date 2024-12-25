import sys
import os
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from sympy import print_tree


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.DeepQ.DQNController import DQNController
from controller.DeepQ.DQN_model import DQN
from rl_env.WRSN import WRSN
print("MPS available:", torch.backends.mps.is_available())

torch.backends.cudnn.benchmark = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def log(net, mcs, visualization_interval=100):
    """
    Logging function to visualize network state periodically.

    Args:
        net: Network object
        mcs: Mobile Chargers list
        visualization_interval: Interval to visualize and log network state
    """
    image_dir = os.path.join('image', 'test_image')
    os.makedirs(image_dir, exist_ok=True)  # Đảm bảo thư mục tồn tại

    next_logging_time = visualization_interval  # Khởi tạo thời điểm ghi log tiếp theo

    while True:
        try:
            if net.env.now >= next_logging_time:
                # Tạo figure và trục để vẽ
                fig, ax = plt.subplots(figsize=(21, 14))
                ax.set_title(f'Network State at Time Step {net.env.now:.2f}')

                # Lấy vị trí và mức năng lượng của cảm biến
                sensor_x = [node.location[0] for node in net.listNodes]
                sensor_y = [node.location[1] for node in net.listNodes]
                sensor_energy = [node.energy for node in net.listNodes]

                # Vẽ cảm biến với màu sắc thể hiện mức năng lượng
                scatter_sensors = ax.scatter(sensor_x, sensor_y, c=sensor_energy, cmap='RdYlGn',
                                             vmin=0, vmax=100, s=100, edgecolor='black', linewidth=1, label='Sensors')
                cbar_sensors = fig.colorbar(scatter_sensors, ax=ax, label='Sensor Energy (%)')

                # Thêm mức năng lượng trên ký hiệu cảm biến
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01  # Điều chỉnh khoảng cách theo trục y
                for x, y, energy in zip(sensor_x, sensor_y, sensor_energy):
                    ax.text(x, y + y_offset, f"{energy:.1f}", fontsize=8, ha='center', va='bottom', color='black')

                # Vẽ trạm cơ sở
                ax.scatter(net.baseStation.location[0], net.baseStation.location[1],
                           color='black', marker='s', s=200, label='Base Station')

                # Lấy vị trí và mức năng lượng của MC
                mc_x = [mc.location[0] for mc in mcs]
                mc_y = [mc.location[1] for mc in mcs]
                mc_energy = [mc.energy for mc in mcs]

                # Vẽ MC với màu sắc thể hiện mức năng lượng
                scatter_mcs = ax.scatter(mc_x, mc_y, c=mc_energy, cmap='Blues',
                                         vmin=0, vmax=100, marker='^', s=300, edgecolor='black', linewidth=1, label='MCs')
                cbar_mcs = fig.colorbar(scatter_mcs, ax=ax, label='MC Energy (%)')

                # Thêm mức năng lượng trên ký hiệu MC
                for x, y, energy in zip(mc_x, mc_y, mc_energy):
                    ax.text(x, y + y_offset, f"{energy:.1f}", fontsize=10, ha='center', va='bottom', color='white', weight='bold')

                # Vẽ vị trí sạc
                charging_x = [loc.charging_location[0] for loc in net.listChargingLocations]
                charging_y = [loc.charging_location[1] for loc in net.listChargingLocations]
                ax.scatter(charging_x, charging_y, color='orange', marker='x', s=100, label='Charging Locations')

                # Cài đặt chi tiết biểu đồ
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.grid(True, linestyle='--', linewidth=0.5)
                ax.legend(loc='upper right')

                # Điều chỉnh margin để không cắt mất chú thích
                ax.margins(0.1)

                # Thêm thông tin thống kê năng lượng
                energy_stats = (f"Simulation Time: {net.env.now:.2f}\n"
                                f"Alive Sensors: {sum(1 for node in net.listNodes if node.status == 1)}/{len(net.listNodes)}\n"
                                f"Avg Sensor Energy: {np.mean([node.energy for node in net.listNodes]):.2f}\n"
                                f"Min Sensor Energy: {np.min([node.energy for node in net.listNodes]):.2f}\n"
                                f"MC Energies: {[f'{mc.energy:.2f}' for mc in mcs]}")
                ax.text(0.99, 0.01, energy_stats, transform=ax.transAxes,
                        horizontalalignment='right', verticalalignment='bottom', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Lưu hình ảnh
                filename = os.path.join(image_dir, f'network_state_{int(net.env.now)}.png')
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved image: {filename}")

                # Cập nhật thời điểm ghi log tiếp theo
                next_logging_time += visualization_interval

            # Tiếp tục bước mô phỏng tiếp theo
            yield net.env.timeout(1.0)

        except Exception as e:
            print(f"Visualization error at step {net.env.now}: {e}")
            # Tiếp tục bước mô phỏng tiếp theo ngay cả khi có lỗi
            yield net.env.timeout(1.0)

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
action_dim = len(network.net.listChargingLocations)  # +1 for the base station action

num_agents  = 1
# Initialize the DQNController
controller = DQNController(num_agents=1, state_dim=state_dim, action_dim=action_dim, config=config)
controller.load_model(agent_id=0, path="save_models/agent_0_model.pth")

model = [DQN(state_dim, action_dim).to(device) for _ in range(num_agents)]
model[0].load_state_dict(torch.load("save_models/agent_0_model_400.pth"))
model[0].eval()

# Reset the environment to start the evaluation
request = network.reset()
log_generator = log(network.net, network.agents, visualization_interval=100)

while not request["terminal"]:
    # print("Agent ID:", request["agent_id"], "Action:", request["action"], "Terminal:", request["terminal"])
    if request["agent_id"] is not None:
        # Extract the flattened state for the agent
        state_flat, embedding_dim = request["state"]
        # Convert state_flat to torch tensor if necessary
        if not isinstance(state_flat, torch.Tensor):
            state_flat = torch.tensor(state_flat, dtype=torch.float32)
        # Select an action using the loaded model
        action = controller.select_action_test(request["agent_id"],model, state_flat, device)
        print("Action:", action)
        # Step the environment with the selected action
        request = network.step(request["agent_id"], action)
    next(log_generator)
    print(network.net.env.now)
    # print("State")
    # print(request["state"])
    # print("Energy")
    # print(network.get_energy(device))

# Print the total simulation time
print("Total simulation time:", network.net.env.now)