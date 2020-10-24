import numpy as np
import gym
import os
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# End/Ground state scores
SOLVED = 200
NEGATIVE_REWARD = 0
CRASH_REWARD = -100
NEGATIVE_LAST_REWARD = 0
REST_REWARD = 100


class LunarLander(object):
    def __init__(self, model, env, name, test_threshold, min_memory=2**6):
        self._env_ = env
        self._min_memory_ = min_memory
        self.name = name

        # Parameters of the environment
        self._n_inputs_ = self._env_.observation_space.shape
        self._n_output_ = self._env_.action_space.n
        self._n_actions_ = self._env_.action_space.n
        self.test_threshold = test_threshold
        self._model_ = model

        # If inf then not solved yet
        self.episodes_till_solved = float('inf')

        # Counter of training episodes
        self.episodes = 0

        # numpy arrays to save history
        self.replay_memory = dict()
        self.replay_memory['state'] = np.empty((0, 8), dtype=np.float)
        self.replay_memory['next_state'] = np.empty((0, 8), dtype=np.float)
        self.replay_memory['step'] = np.empty((0), dtype=np.uint16)
        self.replay_memory['action'] = np.empty((0), dtype=np.uint8)
        self.replay_memory['reward'] = np.empty((0), dtype=np.float)
        self.replay_memory['done'] = np.empty((0), dtype=np.bool)
        self.replay_memory['episode'] = np.empty((0), dtype=np.uint16)
        self.replay_memory['len'] = 0

    # Save step into history arrays
    def add_history(self, **kwargs):
        for key, val in kwargs.items():
            # If the value is array, the value should be stacked, appended otherwise
            if 'state' in key:
                self.replay_memory[key] = np.vstack(
                    [self.replay_memory[key], val])
            else:
                self.replay_memory[key] = np.append(
                    self.replay_memory[key], val)

        self.replay_memory['len'] += 1

    # Fit model to batch num_learn times
    def learn(self, **kwargs):
        for i in range(kwargs['num_learn']):
            states, targets = self._get_batch_(**kwargs)
            self._model_.fit(
                states, targets, epochs=kwargs['epochs'], verbose=0)

    # Get batch
    def _get_batch_(self, **kwargs):
        # if no memory, return nothing
        if self.replay_memory['len'] == 0:
            return None

        # Check if replay memory to small
        if self.replay_memory['len'] < kwargs['lookback_depth']:
            lookback_depth = self.replay_memory['len']
        else:
            lookback_depth = kwargs['lookback_depth']

        # Get look back depth memory
        state = self.replay_memory['state'][-lookback_depth:-1]
        next_state = self.replay_memory['next_state'][-lookback_depth:-1]
        action = self.replay_memory['action'][-lookback_depth:-1]
        reward = self.replay_memory['reward'][-lookback_depth:-1]
        done = self.replay_memory['done'][-lookback_depth:-1]

        # Check if enough for batch
        if kwargs['batch_size'] >= len(state):
            indexes = np.arange(self.replay_memory['len'] - 1)
        else:
            # Get random sample indexes from memory
            indexes = np.random.choice(
                len(state), kwargs['batch_size'] - 1, replace=False)

        # Get batch from memory
        state = np.vstack(
            [state[indexes], self.replay_memory['state'][-1].reshape(1, -1)])
        next_state = np.vstack(
            [next_state[indexes], self.replay_memory['next_state'][-1].reshape(1, -1)])
        action = np.append(action[indexes], self.replay_memory['action'][-1])
        reward = np.append(reward[indexes], self.replay_memory['reward'][-1])
        done = np.append(done[indexes], self.replay_memory['done'][-1])

        # Predict rewards using a model
        h_rewards = self._model_.predict(state)

        # Add rewards
        h_rewards[np.arange(len(h_rewards)), action] = reward
        h_rewards[~done, action[~done]] += kwargs['gamma'] * \
            np.max(self._model_.predict(next_state[~done]), axis=1)

        return state, h_rewards

    # Output current status
    def print_result(self, **kwargs):
        rewards = kwargs["rewards"]
        if rewards <= 0:
            ind = ' - '
        elif rewards >= SOLVED:
            ind = ' * '
        else:
            ind = ' + '
        print(
            f'{self.name} {ind} {self.episodes} {int(rewards)} {kwargs["last_reward"]:.0f} {kwargs["eps"]:.3f}')

    def train(self, **kwargs):
        # Arrays to save rewards and last rewards
        rewards = np.empty((0), dtype=np.float)
        last_rewards = np.empty((0), dtype=np.float)

        for i in range(kwargs['num_episodes']):
            kwargs['eps'] *= kwargs['epsilon_decay']
            episode_rewards, r = self.execute_episode(**kwargs)

            # update counters of agent learning
            self.episodes += 1
            rewards = np.append(rewards, episode_rewards)
            last_rewards = np.append(last_rewards, r)

            if kwargs['verbose']:
                self.print_result(rewards=episode_rewards,
                                  last_reward=r, **kwargs)

            # Game is solved if 100 consecutive episodes average at least 200
            if len(rewards) > 99 and np.mean(rewards[-100:]) > SOLVED - 1:
                self.episodes_till_solved = self.episodes
                break

        return rewards, last_rewards

    def execute_episode(self, **kwargs):
        s = self._env_.reset()
        done = False
        episode_rewards = 0
        step = 1

        # Run 1 episode
        while not done:
            reshaped_s = s.reshape(1, -1)
            s_pred = self._model_.predict(reshaped_s)[0]

            # Generate action
            if np.random.uniform() < kwargs['eps']:
                a = np.random.randint(0, self._n_actions_)
            else:
                a = np.argmax(s_pred)

            # Execute action
            next_s, r, done, _ = self._env_.step(a)

            # Add to experience replay
            self.add_history(
                step=step,
                state=s,
                action=a,
                reward=r,
                next_state=next_s,
                done=done
            )

            s = next_s

            # update counters of the episode
            episode_rewards += r
            step += 1
            self.replay_memory['episode'] += 1

            # Neural Network Learning
            if self.replay_memory['len'] >= self._min_memory_:
                self.learn(**kwargs)

        return episode_rewards, r

    def test(self, num_episodes):
        # Initialize arrays to save rewards and last rewards
        rewards = np.empty((0), dtype=np.float)
        last_rewards = np.empty((0), dtype=np.float)

        for i in range(num_episodes):
            s = self._env_.reset()
            done = False
            episode_rewards = 0
            step = 1

            while not done:
                reshaped_s = s.reshape(1, -1)
                s_pred = self.self._model_.predict(reshaped_s)[0]
                a = np.argmax(s_pred)
                s, r, done, _ = self._env_.step(a)
                episode_rewards += r
                step += 1

            rewards = np.append(rewards, episode_rewards)
            last_rewards = np.append(last_rewards, r)

        return rewards, last_rewards
