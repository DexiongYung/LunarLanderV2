import numpy as np
import gym
import os
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# threshold scores
SUPREWARD = 200
NEGREWARD = 0
SUPNEGLASTREWARD = -100
NEGLASTREWARD = 0
SUPLASTREWARD = 100


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
        self.episodes_till_solved = float('inf')

        # Counter of training episodes
        self.episodes = 0

        # numpy arrays to save history
        self.history = dict()
        self.history['state'] = np.empty((0, 8), dtype=np.float)
        self.history['next_state'] = np.empty((0, 8), dtype=np.float)
        self.history['step'] = np.empty((0), dtype=np.uint16)
        self.history['action'] = np.empty((0), dtype=np.uint8)
        self.history['reward'] = np.empty((0), dtype=np.float)
        self.history['done'] = np.empty((0), dtype=np.bool)
        self.history['episode'] = np.empty((0), dtype=np.uint16)
        self.history['len'] = 0

    # save the training iteration into history arrays
    def add_history(self, **kwargs):
        for key, val in kwargs.items():
            # if the value is array, the value should be stacked, appended otherwise
            if 'state' in key:
                self.history[key] = np.vstack([self.history[key], val])
            else:
                self.history[key] = np.append(self.history[key], val)

        self.history['len'] += 1

    # make a prediction
    def predict(self, state):
        return self._model_.predict(state)

    # train model
    def fit(self, states, targets, epochs=1):
        return self._model_.fit(states, targets, epochs=epochs, verbose=0)

    # learn n-times
    def learn(self, **kwargs):
        for i in range(kwargs['num_learn']):
            states, targets = self._get_batch_(**kwargs)
            self.fit(states, targets, kwargs['epochs'])
    
    # get batch
    def _get_batch_(self, **kwargs):
        # if no history, return nothing
        if self.history['len'] == 0:
            return None

        # check if the history to small
        if self.history['len'] < kwargs['lookback_depth']:
            lookback_depth = self.history['len']
        else:
            lookback_depth = kwargs['lookback_depth']

        # get lookback_depthed history
        state = self.history['state'][-lookback_depth:-1]
        next_state = self.history['next_state'][-lookback_depth:-1]
        action = self.history['action'][-lookback_depth:-1]
        reward = self.history['reward'][-lookback_depth:-1]
        done = self.history['done'][-lookback_depth:-1]

        # check if batch size is too small
        if kwargs['batch_size'] >= len(state):
            indexes = np.arange(self.history['len'] - 1)
        else:
            # get random indexes from the lookback_depthed history
            indexes = np.random.choice(
                len(state), kwargs['batch_size'] - 1, replace=False)

        # get the batch from the lookback_depthed history
        state = np.vstack(
            [state[indexes], self.history['state'][-1].reshape(1, -1)])
        next_state = np.vstack(
            [next_state[indexes], self.history['next_state'][-1].reshape(1, -1)])
        action = np.append(action[indexes], self.history['action'][-1])
        reward = np.append(reward[indexes], self.history['reward'][-1])
        done = np.append(done[indexes], self.history['done'][-1])

        # Predict rewards using a model
        h_rewards = self._model_.predict(state)

        # Add rewards
        h_rewards[np.arange(len(h_rewards)), action] = reward
        h_rewards[~done, action[~done]] += kwargs['gamma'] * \
            np.max(self._model_.predict(next_state[~done]), axis=1)

        return state, h_rewards

    # output statistic
    def rec_statistic(self, rewards, last_reward, **kwargs):
        if rewards <= 0:
            ind = ' - '
        elif rewards >= SUPREWARD:
            ind = ' * '
        else:
            ind = ' + '
        print(
            f'{self.name} {ind} {self.episodes} {int(rewards)} {last_reward:.0f} {kwargs["eps"]:.3f}')

        # Do tests run with eps equal to zero
        if not (self.episodes % self.test_threshold):
            # get rewards
            rewards, last_rewards = self.train(num_episodes=10, eps=0, min_eps=0,
                                               epsilon_decay=0, verbose=False, num_learn=0)

            # calculating histograms for rewards according threshold scores
            hist_rewards, _ = np.histogram(rewards,
                                           bins=[-np.inf, NEGREWARD+np.finfo(np.float16).eps, SUPREWARD, np.inf])

            # calculating histograms for last rewards according threshold scores
            hist_last_rewards, _ = np.histogram(
                last_rewards,
                bins=[-np.inf, SUPNEGLASTREWARD + np.finfo(np.float16).eps,
                      NEGLASTREWARD+np.finfo(np.float16).eps, SUPLASTREWARD, np.inf]
            )

            # output the result of tests
            print(
                f'{self.name} TEST {self.episodes}: '
                f'{hist_rewards} '
                f'{np.min(rewards):.0f} '
                f'{np.max(rewards):.0f} '
                f'{np.median(rewards):.0f} '
                f'{hist_last_rewards} '
                f'{np.min(last_rewards):.0f} '
                f'{np.max(last_rewards):.0f} '
                f'{np.median(last_rewards):.0f}'
            )

    # train agent
    def train(self, num_episodes=2000, **kwargs):
        # arrays to save rewards and last rewards
        rewards = np.empty((0), dtype=np.float)
        last_rewards = np.empty((0), dtype=np.float)

        for i in range(num_episodes):
            kwargs['eps'] *= kwargs['epsilon_decay']
            s = self._env_.reset()
            done = False
            episode_rewards = 0
            step = 1
            while not done:
                reshaped_s = s.reshape(1, -1)

                s_pred = self.predict(reshaped_s)[0]

                # choose the action
                flip = np.random.random()
                if flip < kwargs['eps'] or flip < kwargs['min_eps']:
                    a = np.random.randint(0, self._n_actions_)
                else:
                    a = np.argmax(s_pred)

                # do it
                next_s, r, done, _ = self._env_.step(a)

                # add to the history
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
                self.history['episode'] += 1

                # do some learning
                if self.history['len'] >= self._min_memory_:
                    self.learn(**kwargs)

                # Game is solved if 100 consecutive episodes average at least 200
                if len(last_rewards) > 99 and np.mean(last_rewards[-100:]) > SUPLASTREWARD - 1:
                    self.episodes_till_solved = self.episodes
                    break

            # update counters of agent learning
            self.episodes += 1
            rewards = np.append(rewards, episode_rewards)
            last_rewards = np.append(last_rewards, r)

            if kwargs['verbose']:
                self.rec_statistic(episode_rewards, r, **kwargs)

        return rewards, last_rewards

    def test(self, num_episodes):
        # arrays to save rewards and last rewards
        rewards = np.empty((0), dtype=np.float)
        last_rewards = np.empty((0), dtype=np.float)

        for i in range(num_episodes):
            s = self._env_.reset()
            done = False
            episode_rewards = 0
            step = 1
            while not done:
                reshaped_s = s.reshape(1, -1)

                s_pred = self.predict(reshaped_s)[0]
                a = np.argmax(s_pred)

                # Execute action
                s, r, done, _ = self._env_.step(a)

                # update counters of the episode
                episode_rewards += r
                step += 1

            rewards = np.append(rewards, episode_rewards)
            last_rewards = np.append(last_rewards, r)

        return rewards, last_rewards
