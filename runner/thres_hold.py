import math

param = {
    "gamma": 0.95,
    "learning_rate": 5e-4,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 1000,
    "target_update": 10,
    "batch_size": 128,
    "replay_buffer_capacity": 100000
}
eps_threshold = param.get("epsilon_end", 0.01) + (param.get("epsilon_start", 1.0) - param.get("epsilon_end", 0.01)) * \
                math.exp(-1. * 0 / param.get("epsilon_decay", 100))
for episode in range(1000):
    eps_threshold = param.get("epsilon_end", 0.01) + (param.get("epsilon_start", 1.0) - param.get("epsilon_end", 0.01)) * \
                    math.exp(-1. * episode / param.get("epsilon_decay", 100))
    print(f"Episode {episode+1}/{1000} completed! Epsilon: {eps_threshold}")
