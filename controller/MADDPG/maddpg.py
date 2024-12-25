import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .ReplayBuffer import ReplayBuffer
from .Agent import DDPGAgent

class MADDPG:
    """
    Lớp chính quản lý nhiều Agent DDPG, áp dụng MADDPG logic.
    """
    def __init__(
        self,
        num_agents,
        state_dim,
        action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        device="cpu"
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Tạo list các agent
        self.agents = [
            DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                device=device
            )
            for _ in range(num_agents)
        ]

    def get_actions(self, states, noise_scale=0.0):
        """
        Lấy actions từ tất cả agents. 
        - states shape: (num_agents, state_dim) 
          hoặc (state_dim,) nếu state là global, tùy cách cài đặt
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.get_action(states[i], noise_scale)
            actions.append(action)
        return actions

    def update(self, replay_buffer, batch_size=64):
        """
        Huấn luyện tất cả các agent dựa trên mẫu random từ ReplayBuffer.
        """
        if len(replay_buffer) < batch_size:
            return

        # Lấy batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Mỗi agent update Critic + Actor
        all_next_actions = []
        with torch.no_grad():
            # Lấy action của tất cả agents ở state kế tiếp (từ target_actor)
            for i, agent in enumerate(self.agents):
                next_action_i = agent.target_actor(next_states)
                all_next_actions.append(next_action_i)
            all_next_actions = torch.stack(all_next_actions, dim=1) # shape (bs, num_agents, action_dim)
        
        # Update từng agent
        for i, agent in enumerate(self.agents):
            # critic update
            # -----------------------------------------------------------
            # Tính Q_target = r_i + gamma * Q'(s_{t+1}, a_{t+1})
            # r_i là reward của agent i
            # Q' được tính từ target critic của agent i
            # a_{t+1} = [a1', a2', ... aN'] (tức all_next_actions)
            
            current_states = states  # shape (bs, state_dim)
            current_actions = actions.clone()  # shape (bs, num_agents, action_dim)
            
            # Tạo input cho critic agent i
            agent_action = current_actions[:, i, :]  # action của agent i, shape (bs, action_dim)
            
            Q_current = agent.critic(current_states, agent_action).squeeze(-1)

            # Tính Q_target
            next_action_i = all_next_actions[:, i, :]  # shape (bs, action_dim)
            Q_next = agent.target_critic(next_states, next_action_i).squeeze(-1)
            
            r_i = rewards[:, i]  # reward cho agent i, shape (bs,)
            done_i = dones[:, i] # done cho agent i, shape (bs,)
            
            Q_target = r_i + (1.0 - done_i) * self.gamma * Q_next

            # Critic loss = MSE(Q_current, Q_target)
            critic_loss = F.mse_loss(Q_current, Q_target.detach())

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # actor update
            # -----------------------------------------------------------
            # Mục tiêu: maximize Q (hoặc minimize -Q)
            # Lấy action output từ actor "chính" (ko phải target)
            new_action_i = agent.actor(current_states)
            actor_loss = -agent.critic(current_states, new_action_i).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # soft update target networks
            agent.soft_update(agent.target_actor, agent.actor)
            agent.soft_update(agent.target_critic, agent.critic)
        
    def save(self, save_path):
        """
        Lưu lại toàn bộ agents (actor, critic).
        """
        checkpoint = {}
        for i, agent in enumerate(self.agents):
            checkpoint[f"agent_{i}_actor"] = agent.actor.state_dict()
            checkpoint[f"agent_{i}_critic"] = agent.critic.state_dict()
        torch.save(checkpoint, save_path)

    def load(self, load_path):
        """
        Load lại toàn bộ agents (actor, critic).
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint[f"agent_{i}_actor"])
            agent.critic.load_state_dict(checkpoint[f"agent_{i}_critic"])
            # Đồng bộ sang target
            agent.hard_update(agent.target_actor, agent.actor)
            agent.hard_update(agent.target_critic, agent.critic)