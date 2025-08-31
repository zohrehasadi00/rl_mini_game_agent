from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Simple FIFO replay buffer.
    Stores tuples (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, s, a, r, s2, d):
        i = self.idx
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = d
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int):
        assert self.size >= batch_size, "Not enough samples in buffer"
        ids = np.random.randint(0, self.size, size=batch_size)
        s  = torch.from_numpy(self.states[ids]).to(self.device)
        a  = torch.from_numpy(self.actions[ids]).to(self.device)
        r  = torch.from_numpy(self.rewards[ids]).to(self.device)
        s2 = torch.from_numpy(self.next_states[ids]).to(self.device)
        d  = torch.from_numpy(self.dones[ids]).to(self.device)
        return s, a, r, s2, d
