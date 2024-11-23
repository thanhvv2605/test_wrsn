import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Khởi tạo DQN.

        Tham số:
        - state_dim: Số chiều của trạng thái đầu vào.
        - action_dim: Số lượng hành động (chiều đầu ra).
        - hidden_dim: Số lượng neuron trong lớp ẩn.
        """
        super(DQN, self).__init__()

        # Lớp đầu vào đến lớp ẩn
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Lớp ẩn thứ hai
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Lớp đầu ra dự đoán Q-value cho mỗi hành động
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Truyền tiến của mạng Q-Network.

        Tham số:
        - x: Tensor đầu vào đại diện cho trạng thái.

        Trả về:
        - Q-values cho mỗi hành động (tensor với kích thước [batch_size, action_dim]).
        """
        x = F.relu(self.fc1(x))  # Lớp ẩn thứ nhất với ReLU
        x = F.relu(self.fc2(x))  # Lớp ẩn thứ hai với ReLU
        x = self.fc3(x)  # Lớp đầu ra
        return x