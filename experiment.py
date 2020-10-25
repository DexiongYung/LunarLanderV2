import gym
import json
import argparse

from my_range import Range
from lunar_lander import LunarLander
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--lookback_depth',
                    help='Depth of lookback for experience replay', type=int, default=30000)
parser.add_argument(
    '--num_learn', help='Number of independent batches for fitting model each step', type=int, default=1)
parser.add_argument(
    '--epochs', help='Number of epochs for model', type=int, default=1)
parser.add_argument(
    '--save_dir', help='Directory for saving plot and stats', type=str, default='./data')
parser.add_argument('--test_threshold',
                    help='Number of episodes before printing and test', type=int, default=200)
parser.add_argument(
    '--model_ll_1', help='Width of linear layer 1', type=int, default=128)
parser.add_argument(
    '--model_ll_2', help='Width of linear layer 2', type=int, default=64)
parser.add_argument('--max_num_episodes',
                    help='Max number of episodes', type=int, default=2000)
parser.add_argument('--max_test_episodes',
                    help='Max test episodes', type=int, default=100)
parser.add_argument(
    '--epsilon', help='Percent chance of exploration', type=float, choices=[Range(0.0, 1.0)], default=1.0)
parser.add_argument(
    '--epsilon_decay', help='Decay of epsilon as Q is learned', type=float, choices=[Range(0.0, 1.0)], default=0.998)
parser.add_argument(
    '--gamma', help='Future discount rate', type=float, choices=[Range(0.0, 1.0)], default=1)
parser.add_argument(
    '--is_verbose', help='Should print test results?', type=bool, default=True)
parser.add_argument(
    '--LR', help='Learning rate of model', type=float, default=0.0001)
args = parser.parse_args()


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    num_learn = args.num_learn
    epoch = args.epochs
    lookback_depth = args.lookback_depth
    batch_size = args.batch_size
    num_episodes = args.max_num_episodes
    eps = args.epsilon
    eps_decay = args.epsilon_decay
    is_verbose = args.is_verbose
    gamma = args.gamma
    test_threshold = args.test_threshold
    model_ll_1 = args.model_ll_1
    model_ll_2 = args.model_ll_2
    name = f'lunar_lander_gamma{gamma}'
    max_test_ep = args.max_test_episodes
    save_dir = args.save_dir

    # Create Model
    model = Sequential()
    model.add(
        Dense(model_ll_1, input_shape=env.observation_space.shape, activation='relu'))
    model.add(Dense(model_ll_2, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.LR))

    # Create Agent
    ll = LunarLander(model=model, env=env, name=name,
                     test_threshold=test_threshold)

    # Begin training
    train_rewards, train_last_rewards = ll.train(
        num_episodes=num_episodes,
        num_learn=num_learn,
        epochs=epoch,
        eps=eps,
        epsilon_decay=eps_decay,
        verbose=is_verbose,
        lookback_depth=lookback_depth,
        batch_size=batch_size,
        gamma=gamma,
    )

    # Begin Test
    test_rewards, test_last_rewards = ll.test(max_test_ep)

    num_episodes = ll.episodes_till_solved

    data = {}
    data['episodes'] = num_episodes
    data['train rewards'] = train_rewards
    data['train_last_rewards'] = train_last_rewards
    data['test rewards'] = test_rewards
    data['test_last_rewards'] = test_last_rewards

    with open(f'{save_dir}/{name}.txt', 'w') as outfile:
        json.dump(data, outfile)
