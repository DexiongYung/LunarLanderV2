import gym
import os
import json
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from replay_buffer import ReplayBuffer
from Q_network import QNetwork
from constant import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument(
    '--replay_size', help='Buffer size of replay', type=int, default=1e5)
parser.add_argument('--max_num_episodes',
                    help='Max number of episodes', type=int, default=5000)
parser.add_argument('--max_test_episodes',
                    help='Max test episodes', type=int, default=100)
parser.add_argument('--max_episode_steps',
                    help='Max steps per episodes', type=int, default=1000)
parser.add_argument(
    '--epsilon', help='Percent chance of exploration', type=float, default=1.0)
parser.add_argument(
    '--min_epsilon', help='Min percent chance of exploration', type=float, default=0.01)
parser.add_argument(
    '--decay', help='Decay of epsilon as Q is learned', type=float, default=0.998)
parser.add_argument(
    '--gamma', help='Future discount rate', type=float, default=1)
parser.add_argument(
    '--LR', help='Learning rate of model', type=float, default=0.0001)
parser.add_argument(
    '--Tau', help='Soft update parameter', type=float, default=1e-3)
parser.add_argument(
    '--step_update', help='Number of steps before update', type=int, default=4)
args = parser.parse_args()

BUFFER_SIZE = int(args.replay_size)  # Replay memory size
BATCH_SIZE = args.batch_size         # Number of experiences to sample from memory
GAMMA = args.gamma      # Discount factor
TAU = args.Tau              # Soft update parameter for updating fixed q network
LR = args.LR               # Q Network learning rate
UPDATE_EVERY = args.step_update        # How often to update Q network
MAX_EPISODES = args.max_num_episodes  # Max number of episodes to play
MAX_TEST_EPISODES = args.max_test_episodes  # Max number of test episodes
MAX_STEPS = args.max_episode_steps  # Max steps allowed in a single episode/play
EPS_START = args.epsilon     # Default/starting value of eps
EPS_DECAY = args.decay    # Epsilon decay rate
EPS_MIN = args.min_epsilon       # Minimum epsilon


class DQNAgent:
    def __init__(self, name, env, seed):
        """
        DQN Agent interacts with the environment, 
        stores the experience and learns from it

        Parameters
        ----------
        state_size (int): Dimension of state
        action_size (int): Dimension of action
        seed (int): random seed
        """
        self.name = name
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.seed = random.seed(seed)
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(self.state_size, self.action_size, seed).to(DEVICE)
        self.fixed_network = QNetwork(self.state_size, self.action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0

    def update_memory(self, state, action, reward, next_state, done):
        """
        Update Agent's knowledge

        Parameters
        ----------
        state (array_like): Current state of environment
        action (int): Action taken in current state
        reward (float): Reward received after taking action 
        next_state (array_like): Next state returned by the environment after taking action
        done (bool): whether the episode ended after taking action
        """
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1

    def is_time_to_learn(self):
        return self.timestep % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE

    def learn(self, experiences):
        """
        Learn from experience by training the q_network 

        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)

    def update_fixed_network(self, q_network, fixed_network):
        """
        Update fixed network by copying weights from Q network using TAU param

        Parameters
        ----------
        q_network (PyTorch model): Q network
        fixed_network (PyTorch model): Fixed target network
        """
        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(
                TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data)

    def get_action(self, state, eps: float = 0.0):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            # set the network into evaluation mode
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)
        eps = EPS_START
        num_episodes = MAX_EPISODES
        for episode in range(1, MAX_EPISODES + 1):
            state = self.env.reset()
            score = 0
            for t in range(MAX_STEPS):
                action = self.get_action(state, eps)
                next_state, reward, done, info = self.env.step(action)
                self.update_memory(state, action, reward, next_state, done)

                if self.is_time_to_learn():
                    sampled_experiences = self.memory.sample()
                    self.learn(sampled_experiences)

                state = next_state
                score += reward
                if done:
                    break

                eps = max(eps * EPS_DECAY, EPS_MIN)

            scores.append(score)
            self.checkpoint(f'weights/{self.name}.pth')

            is_solved = np.mean(scores_window) >= 200

            if is_solved:
                num_episodes = episode
                break
            else:
                scores_window.append(score)

        return num_episodes, scores

    def test(self, num_episodes, num_steps: int = MAX_STEPS):
        scores = []
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(num_steps):
                action = self.get_action(state, 0)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                score += reward

                if done:
                    break

            scores.append(score)

        return scores

    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)


name = f'g{GAMMA}_lr{LR}_decay_{EPS_DECAY}'
env = gym.make('LunarLander-v2')

for file in os.listdir('weights'):
    print(f'Running {file}')
    dqn_agent = DQNAgent(name, env, seed=0)
    dqn_agent.q_network.load_state_dict(torch.load(f'weights/{file}',map_location=torch.device('cpu')))
    scores = dqn_agent.test(MAX_TEST_EPISODES)
    name = file.replace('pth','')
    data = {}
    data['scores'] = scores
    with open(f'test/{name}.txt', 'w') as outfile:
        json.dump(data, outfile)

