import os
import argparse
import numpy as np
import gym
import d3rlpy
from d3rlpy.algos import DQN, DiscreteBCQ, DiscreteCQL
from d3rlpy.dataset import MDPDataset

from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import evaluate_on_environment, average_value_estimation_scorer


def train_offline_poilcy(epochs, dataset_name, save_dir):
    # if the input_dir does not exist, we create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for now
    assert dataset_name in ['cartpole-replay', 'cartpole-random', 'acrobot-replay', 'acrobot-random']

    if 'acrobot' not in dataset_name:
        dataset, env = d3rlpy.datasets.get_dataset(dataset_name)
    else:
        # we load in differently
        if 'replay' in dataset_name:
            data_path = os.path.join(save_dir, 'acrobot_v1_replay_dataset.h5')
        else:
            data_path = os.path.join(save_dir, 'acrobot_v1_random_dataset.h5')
        dataset = MDPDataset.load(data_path)
        env = gym.make('Acrobot-v1')

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.5, random_state=42)

    bcq = DiscreteBCQ()
    bcq.build_with_dataset(dataset)

    evaluate_scorer = evaluate_on_environment(env)
    init_rewards = evaluate_scorer(bcq)
    print("Initial reward: ", init_rewards)

    bcq.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=epochs,
            save_interval=1,  # the task is small, we need to save everything
            logdir=save_dir,
            experiment_name=f'{dataset_name}_bcq_eps_{epochs}',
            scorers={
                'value_scale': average_value_estimation_scorer,
                'environment': evaluate_scorer
            })

    final_reward = evaluate_scorer(bcq)
    print("Final reward: ", final_reward)
    with open(os.path.join(save_dir, f'{dataset_name}_bcq_eps_{epochs}.txt'), 'w') as f:
        f.write(f"Final reward: {final_reward}")

    # save the model to save_dir
    bcq.save_policy(os.path.join(save_dir, f'{dataset_name}_bcq_eps_{epochs}.pt'))


if __name__ == '__main__':
    # write an argument parser that has dataset name and input directory

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cartpole-replay')
    parser.add_argument('--save_dir', type=str, default='model_logs')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    train_offline_poilcy(args.epochs, args.dataset, args.save_dir)
