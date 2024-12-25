import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    """
    Mạng Actor để sinh action từ state.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Khởi tạo trọng số hoặc có thể dùng init mặc định của PyTorch.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        """
        Trả về action (chưa áp dụng hàm kích hoạt ra output nếu cần).
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Ở nhiều bài toán continuous, chúng ta thường dùng torch.tanh()
        # hoặc 1 activation phù hợp cho output
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """
    Mạng Critic để ước lượng Q-value dựa trên (state, action).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Giả sử chúng ta truyền vào state và action ở tầng đầu tiên
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Khởi tạo trọng số.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        """
        Tính Q-value cho cặp (state, action).
        """
        x = torch.relu(self.fc1(torch.cat([state, action], dim=-1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPGAgent:
    """
    Lớp Agent, mỗi Agent có Actor, Critic, target Actor, target Critic riêng.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        device="cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Khởi tạo Actor & Critic
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Khởi tạo Actor Target & Critic Target
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        # Đồng bộ trọng số ban đầu giữa mạng chính và mạng target
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        # Khởi tạo optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def hard_update(self, target, source):
        """
        Copy trực tiếp trọng số từ mạng source -> mạng target.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        """
        Soft update: target = tau * source + (1 - tau) * target
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def get_action(self, state, noise_scale=0.0):
        """
        Lấy action từ Actor. Có thể thêm nhiễu (noise) cho exploration.
        """
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()  # để tránh ảnh hưởng batchnorm/dropout (nếu có)
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()

        # convert to numpy
        action = action.cpu().numpy()
        # thêm noise
        action += noise_scale * np.random.randn(*action.shape)
        # clip nếu cần
        action = np.clip(action, -1.0, 1.0)
        return action

    def update(
        self,
        transitions,
        other_agents,
    ):
        """
        Hàm huấn luyện Critic & Actor cho agent này trong môi trường multi-agent.
        Tạm thời để logic huấn luyện trong hàm update ở file `maddpg.py`.
        Hoặc có thể triển khai ở đây nếu muốn.
        """
        pass