import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Bộ nhớ để lưu trữ trải nghiệm trong môi trường multi-agent.
    """

    def __init__(self, max_size, state_dim, action_dim, num_agents, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Lưu state, next_state dạng chung (của toàn hệ thống)
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)

        # Lưu action, reward, done cho từng agent
        # action_dim là số chiều action cho MỖI agent
        # => tổng action_dim của toàn bộ agent = num_agents * action_dim (nếu cần)
        self.actions = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((max_size, num_agents), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        Thêm trải nghiệm mới vào buffer.
        - state: shape (state_dim, )
        - action: shape (num_agents, action_dim)
        - reward: shape (num_agents, )
        - next_state: shape (state_dim, )
        - done: shape (num_agents, ) hoặc boolean
        """
        idx = self.ptr

        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done.astype(float)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=64):
        """
        Lấy ngẫu nhiên batch_size mẫu từ Replay Buffer.
        Trả về (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        # Convert to Tensor
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size