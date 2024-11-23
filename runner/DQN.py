import sys
import os
import torch
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from controller.deep_q.train_deepq import train_dqn

import numpy as np
from rl_env.WRSN import WRSN
import yaml

def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        print(net.env.now, net.listNodes[0].energy)
        yield net.env.timeout(1.0)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=1)
# Test Upload


# train_dqn(env, device, num_episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10, lr=0.001, replay_buffer_capacity=10000, warm_up_time=100, max_steps=1000, save_path="controller/deep_q/models/dqn_model.pth")
train_dqn(env, num_episodes=1000, input_dim=4, hidden_dim=64, output_dim=3, buffer_size=10000, save_path="save_model/dqn_model.pth")

