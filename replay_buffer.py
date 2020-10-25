import torch
import random
import numpy as np
from constant import DEVICE
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Replay memory allow agent to record experiences and learn from them

        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """ 
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors 
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(np.vstack(
            [experience.state for experience in experiences if experience is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(
            [experience.action for experience in experiences if experience is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(
            [experience.reward for experience in experiences if experience is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(
            [experience.next_state for experience in experiences if experience is not None])).float().to(DEVICE)
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack(
            [experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
