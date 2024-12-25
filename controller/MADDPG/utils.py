import torch

def soft_update(target, source, tau):
    """
    Soft update: target = tau * source + (1 - tau) * target
    """
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

def hard_update(target, source):
    """
    Hard update: copy toàn bộ tham số từ source sang target.
    """
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(s_param.data)

def get_device():
    """
    Trả về 'cuda' nếu có GPU, ngược lại 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class OUNoise:
    """
    Ornstein-Uhlenbeck process cho exploration trong continuous action spaces.
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.mu * torch.ones(self.action_dim)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self.action_dim)
        self.state = x + dx
        return self.state