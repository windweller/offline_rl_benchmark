import os
import argparse
import numpy as np
import d3rlpy
from d3rlpy.algos import DQN

import gym
from d3rlpy.metrics.scorer import evaluate_on_environment, average_value_estimation_scorer


def train_online_collection_poilcy(epochs, dataset_name, save_dir):
    # if the input_dir does not exist, we create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for now
    env = gym.make(dataset_name)
    eval_env = gym.make(dataset_name)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256])

    dqn = d3rlpy.algos.DQN(
        batch_size=128,
        learning_rate=0.00063,
        target_update_interval=250,
        use_gpu=False)

    # experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # exploration strategy
    # in this tutorial, epsilon-greedy policy with static epsilon=0.3
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(1.0, 0.1, 100000)

    # initialize neural networks with the given environment object.
    # this is not necessary when you directly call fit or fit_online method.
    dqn.build_with_env(env)

    evaluate_scorer = evaluate_on_environment(env)
    init_rewards = evaluate_scorer(dqn)
    print("Initial reward: ", init_rewards)

    dqn.fit_online(
        env,
        buffer,
        explorer,
        n_steps=200000,  # 10k steps?
        eval_env=eval_env,
        save_interval=1,  # the task is small, we need to save everything
        experiment_name=f'{dataset_name}_dqn_eps_{epochs}',
        logdir=save_dir)

    final_reward = evaluate_scorer(dqn)
    print("Final reward: ", final_reward)
    with open(os.path.join(save_dir, f'{dataset_name}_dqn_eps_{epochs}.txt'), 'w') as f:
        f.write(f"Final reward: {final_reward}")

    # save the model to save_dir
    dqn.save_policy(os.path.join(save_dir, f'{dataset_name}_dqn_eps_{epochs}.pt'))

    dataset = buffer.to_mdp_dataset()

    # save MDPDataset
    dataset.dump(os.path.join(save_dir, "acrobot_v1_replay_dataset.h5"))

    return dqn


def collect_dataset(save_dir):
    env = gym.make("Acrobot-v1")

    # setup algorithm
    random_policy = d3rlpy.algos.DiscreteRandomPolicy()

    # prepare experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

    # start data collection
    random_policy.collect(env, buffer, n_steps=100000)

    # export as MDPDataset
    dataset = buffer.to_mdp_dataset()

    # save MDPDataset
    dataset.dump(os.path.join(save_dir, "acrobot_v1_random_dataset.h5"))

def collect_n_datasets(save_dir, env_name, n=20):
    from tqdm import tqdm

    env = gym.make(env_name)

    # setup algorithm
    random_policy = d3rlpy.algos.DiscreteRandomPolicy()

    for i in tqdm(range(n)):
        # prepare experience replay buffer
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

        # start data collection
        random_policy.collect(env, buffer, n_steps=100000)

        # export as MDPDataset
        dataset = buffer.to_mdp_dataset()

        # save MDPDataset
        dataset.dump(os.path.join(save_dir, f"{env_name}_random_dataset_s{i}.h5"))


if __name__ == '__main__':
    # write an argument parser that has dataset name and input directory

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='Acrobot-v1')
    # parser.add_argument('--save_dir', type=str, default='model_logs')
    # parser.add_argument('--epochs', type=int, default=10)
    # args = parser.parse_args()
    #
    # train_online_collection_poilcy(args.epochs, args.dataset, args.save_dir)
    # collect_dataset(args.save_dir)

    collect_n_datasets("control_datasets", "Acrobot-v1")
    collect_n_datasets("control_datasets", "CartPole-v1")