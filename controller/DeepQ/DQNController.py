import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque
from controller.DeepQ.DQN_model import DQN
from controller.DeepQ.ReplayBuffer import ReplayBuffer


class DQNController:
    def __init__(self, num_agents, state_dim, action_dim, config, device=torch.device('cpu')):
        # Load hyperparameters from config
        self.device = device
        self.num_agents = num_agents  # Number of agents (mobile chargers)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-3)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 100)
        self.target_update_freq = config.get("target_update", 10)
        self.batch_size = config.get("batch_size", 512)
        self.replay_buffer_capacity = config.get("replay_buffer_capacity", 100000)
        print("REPLAY SIZE", self.replay_buffer_capacity)
        print("BATCH SIZE", self.batch_size)
        # Initialize steps_done for each agent
        self.steps_done = [0 for _ in range(num_agents)]
        self.episode_count = 0  # Episode count

        # Replay buffers for all agents
        self.replay_buffers = [ReplayBuffer(capacity=self.replay_buffer_capacity) for _ in range(num_agents)]

        # Networks for all agents
        self.q_networks = [DQN(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.target_networks = [DQN(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(self.q_networks[i].parameters(), lr=self.lr) for i in range(num_agents)]

        # Sync target networks with Q-networks
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_networks[i].eval()

        # Tạo bộ tạo số ngẫu nhiên cục bộ cho mỗi tác nhân
        self.rng = np.random.default_rng()

    def select_action(self, agent_id, state):
        """
        Lựa chọn hành động với epsilon giảm theo pha.
        """
        # Giai đoạn đầu: exploration chậm
        if self.episode_count < 0.5 * 1000:
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            math.exp(-1. * self.episode_count / (0.5 * self.epsilon_decay))
        # Giai đoạn sau: exploitation nhanh hơn
        else:
            eps_threshold = max(self.epsilon_end,
                                self.epsilon_end * 0.99 ** (self.episode_count - 0.5 * 1000))

        # In ra epsilon để kiểm tra
        rand_value = self.rng.random()
        print(f"Agent {agent_id} - Episode {self.episode_count}, Epsilon: {eps_threshold}, Random Value: {rand_value}")

        if rand_value > eps_threshold:
            # Exploitation
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_networks[agent_id](state_tensor)
            action = int(torch.argmax(q_values).item())

            print(f"Agent {agent_id} - Q-values: {q_values.cpu().numpy()}, Selected action: {action}")
        else:
            # Exploration
            action = self.rng.integers(0, self.action_dim)
            print(f"Agent {agent_id} - Exploration action selected: {action}")
        return action

    def select_action_test(self, agent_id, model, state, device):
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state_tensor = state.to(device).unsqueeze(0)
        with torch.no_grad():
            q_values = model[agent_id](state_tensor)
        print(f"Agent {agent_id} - Q-values: {q_values.cpu().numpy()}")
        action = int(torch.argmax(q_values).item())
        return action


    def store_transition(self, agent_id, state, action, reward, next_state, done):
        # Chuyển dữ liệu sang tensor trên thiết bị
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.int64, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor([done], dtype=torch.float32, device=self.device)

        self.replay_buffers[agent_id].store(state, action, reward, next_state, done)

    def train_agent(self, agent_id):
        """
        Train the Q-network of a specific agent using its replay buffer.
        """
        if len(self.replay_buffers[agent_id]) < 80*self.batch_size:

            print(f"Replay buffer không đủ mẫu: {len(self.replay_buffers[agent_id])}/{80*self.batch_size}")
            return  # Not enough samples to train
        # Sample a mini-batch from the replay buffer
        batch = self.replay_buffers[agent_id].sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # Chuyển đổi sang tensor và chuyển sang thiết bị
        states = torch.stack(states).float().to(self.device)
        actions = torch.cat(actions).long().to(self.device)
        rewards = torch.cat(rewards).float().to(self.device)
        next_states = torch.stack(next_states).float().to(self.device)
        dones = torch.cat(dones).float().to(self.device)

        # Thêm dimension cho actions và rewards nếu cần
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # Q-values of current states
        q_values = self.q_networks[agent_id](states).gather(1, actions)
        print(f"Replay Buffer Size: {len(self.replay_buffers[agent_id])}")

        # Q-values of next states from target network
        with torch.no_grad():
            max_next_q_values = self.target_networks[agent_id](next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values


        # Compute loss and update the agent's Q-network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        print(f"AGENT {agent_id} - Loss: {loss.item()}")
        # print(f"Q-values: {q_values.detach().cpu().numpy()}")
        # print(f"Target Q-values: {target_q_values.detach().cpu().numpy()}")

    def sync_target_network(self, agent_id):
        """
        Update the target network of a specific agent to match its Q-network.
        """
        self.steps_done[agent_id] += 1  # Tăng steps_done
        if self.steps_done[agent_id] % self.target_update_freq == 0:
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())

    def increment_episode(self):
        self.episode_count += 1

    def save_model(self, agent_id, path):
        """
        Save the Q-network of a specific agent to a file.
        """
        torch.save(self.q_networks[agent_id].state_dict(), path)

    def load_model(self, agent_id, path):
        """
        Load the Q-network of a specific agent from a file.
        """
        map_location = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.q_networks[agent_id].load_state_dict(torch.load(path, map_location=map_location))

        # self.q_networks[agent_id].load_state_dict(torch.load(path))

        self.q_networks[agent_id].eval()