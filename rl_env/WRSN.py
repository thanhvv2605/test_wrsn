import yaml
import copy
import gym
import random
from torch_geometric.graphgym.optim import none_scheduler
from rl_env.state_representation.GNN import GCN
from gym import spaces
import numpy as np
import sys
import os
import warnings

root_dir = os.getcwd()
import torch
from scipy.spatial.distance import euclidean

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from rl_env.state_representation.StateRepresentation import GraphRepresentation


class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent, warm_up_time=100, device=torch.device('cpu')):
        self.device = device
        self.scenario_io = NetworkIO(scenario_path)
        with open(agent_type_path, "r") as file:
            self.agent_phy_para = yaml.safe_load(file)
        self.num_agent = num_agent
        self.warm_up_time = warm_up_time
        self.epsilon = 1e-9
        self.agents_process = [None for _ in range(num_agent)]
        self.agents_action = [None for _ in range(num_agent)]
        self.agents_prev_state = [None for _ in range(num_agent)]
        self.agents_prev_fitness = [None for _ in range(num_agent)]
        self.agents_exclusive_reward = [0 for _ in range(num_agent)]
        self.reset()
        for entry in self.net.listChargingLocations:
            print("Charging location: ")
            print(entry.id, entry.charging_location)
        print(len(self.net.listChargingLocations))
        self.action_space = spaces.Discrete(len(self.net.listChargingLocations) + 1)

    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate()) & self.env.process(self.update_reward())
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in
                       range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id
            agent.cur_phy_action = [self.net.baseStation.location[0], self.net.baseStation.location[1], 0]
        self.moving_time_max = (euclidean(np.array([self.net.frame[0], self.net.frame[2]]),
                                          np.array([self.net.frame[1], self.net.frame[3]]))) / self.agent_phy_para[
                                   "velocity"]
        self.charging_time_max = (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe[
            "threshold"]) / (self.agent_phy_para["alpha"] / (self.agent_phy_para["beta"] ** 2))
        self.avg_nodes_agent = (self.net.nodes_density * np.pi * (self.agent_phy_para["charging_range"] ** 2))
        self.env.run(until=self.warm_up_time)
        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True
        for id, agent in enumerate(self.agents):
            state_flat, embedding_dim = self.get_state(agent.id)
            self.agents_prev_state[id] = (state_flat, embedding_dim)
            self.agents_action[id] = 0  # Initialize action
            self.agents_process[id] = self.env.process(
                self.agents[id].operate_step(copy.deepcopy(agent.cur_phy_action)))
            self.agents_exclusive_reward[id] = 0.0

        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id": id,
                        "prev_state": self.agents_prev_state[id],
                        "action": self.agents_action[id],
                        "reward": 0.0,
                        "state": self.agents_prev_state[id],
                        "terminal": tmp_terminal,
                        "info": [self.net, self.agents]}
        return {"agent_id": None,
                "prev_state": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal": tmp_terminal,
                "info": [self.net, self.agents]}

    def config_action(self, agent_id, action):
        if action == len(self.net.listChargingLocations):
            # Return to base station
            x = self.net.baseStation.location[0]
            y = self.net.baseStation.location[1]
            charging_time = self.agent_phy_para['battery_capacity'] / self.agent_phy_para['input_voltage']
        else:
            charging_location = self.net.listChargingLocations[action]
            x = charging_location.charging_location[0]
            y = charging_location.charging_location[1]
            charging_time = self.charging_time_max
        return np.array([x, y, charging_time])

    def update_reward(self):
        """_summary_
        Hàm update_reward nhằm:
        - Xác định mức độ ưu tiên cho việc sạc của các node trong mạng.
        - Tính toán điểm thưởng cho mỗi tác nhân dựa trên sự cải thiện năng lượng của các node mà nó ảnh hưởng.
        - Cập nhật điểm thưởng độc quyền (agents_exclusive_reward) của từng tác nhân, giúp đánh giá hiệu suất của chúng trong việc kéo dài thời gian sống của các node.
        """
        yield self.env.timeout(0)

    def get_state(self, agent_id):
        """
        :param agent_id:
        :return:
        """

        model_path = os.path.join(root_dir, "rl_env", "grap_model.pth")

        data = GraphRepresentation.create_graph(self.net)
        num_features = data.x.size(1)
        num_classes = len(self.net.listChargingLocations) + 1
        hidden_dim = 512
        output_dim = 83  # số lượng lớp đầu ra, ví dụ
        GNN_model = GCN(num_features, hidden_dim, output_dim, num_classes).to(self.device)

        # Tải lại trạng thái của mô hình từ file
        # GNN_model.load_state_dict(torch.load(model_path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GNN_model.load_state_dict(torch.load(model_path, weights_only=True))
        data = data.to(self.device)
        with torch.no_grad():
            _, embeddings = GNN_model(data.x, data.edge_index)

        enegy = self.get_enegy(device=embeddings.device)
        embeddings = torch.cat((embeddings, enegy), 1)
        embeddings_np = embeddings.detach().cpu().numpy()
        embedding_dim = embeddings_np.shape[1]
        embeddings_flat = embeddings_np.flatten()
        return embeddings_flat, embedding_dim

    def get_enegy(self, device):
        arr_energy = []
        for node in self.net.listNodes:
            arr_energy.append(node.energy / self.scenario_io.node_phy_spe["capacity"])
        arr_energy.append(1)
        arr_energy = torch.tensor(arr_energy, device=device)
        tensor_energy = arr_energy.view(-1, 1)
        return tensor_energy

    def get_reward(self, agent_id, prev_state, curr_state):
        prev_state_flat, embedding_dim = prev_state
        curr_state_flat, _ = curr_state

        # Extract energy levels from flattened states
        prev_energy = prev_state_flat[(embedding_dim - 1)::embedding_dim]
        curr_energy = curr_state_flat[(embedding_dim - 1)::embedding_dim]

        # Compute the change in mean energy
        delta_mean_energy = curr_energy.mean() - prev_energy.mean()

        # Compute the change in minimum energy
        delta_min_energy = curr_energy.min() - prev_energy.min()

        reward = delta_mean_energy + delta_min_energy

        print("delta_mean_energy:", delta_mean_energy)
        print("delta_min_energy:", delta_min_energy)
        print("---- reward ----")
        print(reward)
        return reward

    def get_network_fitness(self):
        node_t = [-1 for node in self.net.listNodes]
        tmp1 = []
        tmp2 = []
        for node in self.net.baseStation.direct_nodes:
            if node.status == 1:
                tmp1.append(node)
                if node.energyCS == 0:
                    node_t[node.id] = float("inf")
                else:
                    node_t[node.id] = (node.energy - node.threshold) / (node.energyCS)
        while True:
            if len(tmp1) == 0:
                break
            for node in tmp1:
                for neighbor in node.neighbors:
                    if neighbor.status != 1:
                        continue
                    if neighbor.energyCS == 0:
                        neighborLT = float("inf")
                    else:
                        neighborLT = (neighbor.energy - neighbor.threshold) / (neighbor.energyCS)
                    if node_t[neighbor.id] == -1 or (
                            node_t[node.id] > node_t[neighbor.id] and neighborLT > node_t[neighbor.id]):
                        tmp2.append(neighbor)
                        node_t[neighbor.id] = min(neighborLT, node_t[node.id])

            tmp1 = tmp2[:]
            tmp2.clear()
        target_t = [0 for target in self.net.listTargets]
        for node in self.net.listNodes:
            for target in node.listTargets:
                target_t[target.id] = max(target_t[target.id], node_t[node.id])
        return np.array(target_t)

    def step(self, input_action):
        print("---- step ----")
        print(input_action)
        agent_id = 0  # Assuming a single agent with ID 0
        if input_action is not None:
            action = int(input_action)
            self.agents_action[agent_id] = action
            # Capture previous state before action
            prev_state = self.agents_prev_state[agent_id]
            # Start the agent's action
            self.agents_process[agent_id] = self.env.process(
                self.agents[agent_id].operate_step(self.config_action(agent_id, action))
            )
            self.agents_exclusive_reward[agent_id] = 0
        else:
            prev_state = None

        # Run the environment
        general_process = self.net_process
        for id, agent in enumerate(self.agents):
            if agent.status != 0:
                general_process = general_process | self.agents_process[id]
        self.env.run(until=general_process)

        # After environment runs, get current state
        curr_state = self.get_state(agent_id)
        self.agents_prev_state[agent_id] = curr_state

        if self.net.alive == 0:
            return {
                "agent_id": None,
                "prev_state": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal": True,
                "info": [self.net, self.agents],
            }

        if prev_state is not None:
            # Compute reward based on state changes
            reward = self.get_reward(agent_id, prev_state, curr_state)
        else:
            reward = 0.0

        # Check if agent is ready to act again
        if euclidean(self.agents[agent_id].location, self.agents[agent_id].cur_phy_action[0:2]) < self.epsilon and \
                self.agents[agent_id].cur_phy_action[2] == 0:
            return {
                "agent_id": agent_id,
                "prev_state": prev_state,
                "action": self.agents_action[agent_id],
                "reward": reward,
                "state": curr_state,
                "terminal": False,
                "info": [self.net, self.agents],
            }
        else:
            # Agent is still moving or charging
            return {
                "agent_id": None,
                "prev_state": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal": False,
                "info": [self.net, self.agents],
            }