import random
from collections import deque
import torch


class ReplayBuffer:
    def __init__(self, capacity, similarity_threshold=0.99):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.similarity_threshold = similarity_threshold

    def store(self, state, action, reward, next_state, done):
        # Dữ liệu đã được chuyển sang tensor và thiết bị trong hàm store_transition của DQNController
        print(f"Storing sample: {state}, {action}, {reward}, {next_state}, {done}")
        if not self.is_state_similar(state):
            self.buffer.append((state, action, reward, next_state, done))
        else:
            print("State is too similar to existing samples, skipping addition.")

    def is_state_similar(self, new_state):
        """
        Kiểm tra trạng thái mới có tương tự với các trạng thái trong buffer hay không.
        Args:
            new_state (Tensor): Trạng thái cần kiểm tra.
        Returns:
            bool: True nếu trạng thái tương tự với trạng thái trong buffer.
        """
        for stored_sample in self.buffer:
            stored_state = stored_sample[0]
            print(f"Similarity: {self.cosine_similarity(new_state, stored_state)}")
            if self.cosine_similarity(new_state, stored_state) > self.similarity_threshold:
                return True
        return False

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    @staticmethod
    def cosine_similarity(state1, state2):
        """
        Tính toán cosine similarity giữa hai trạng thái.
        Args:
            state1 (Tensor): Trạng thái thứ nhất.
            state2 (Tensor): Trạng thái thứ hai.
        Returns:
            float: Cosine similarity giữa hai trạng thái.
        """
        state1 = state1.flatten()
        state2 = state2.flatten()
        dot_product = torch.dot(state1, state2)
        norm_state1 = torch.norm(state1)
        norm_state2 = torch.norm(state2)
        return dot_product / (norm_state1 * norm_state2 + 1e-8)

    def __len__(self):
        return len(self.buffer)