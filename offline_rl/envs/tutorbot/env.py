"""
A Fake Gym Env for Sepsis
It will actually not run, but it will return the correct configuration
"""

import gym

import random

import numpy as np
import offline_rl.envs.sepsis.counterfactual as cf
import pandas as pd
import pickle
import os

from offline_rl.envs.tutorbot.tutor_env import StudentEnv
from offline_rl.envs.dataset import ProbabilityMDPDataset, CSVDataset

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

from d3rlpy.dataset import MDPDataset, Transition

from d3rlpy.metrics.scorer import AlgoProtocol

class TutorBot(StudentEnv, CSVDataset):
    def get_feature_names(self):
        return ['pre', 'anxiety', 'thinking', 'last_step']

    def get_action_names(self):
        # for data loading purposes
        return ['p_encourage', 'p_hint', 'p_guided_prompt']

    def get_reward_name(self):
        return "reward"

    def get_trajectory_marking_name(self):
        return "user_id"

    def step(self, action: float):
        """
        We wrap over the parent class' step method, to provide interface to gym...
        :param action: the action to take
        :return: the next state, the reward, whether the episode is done, and the info
        """
        state, reward, done, info = super().step(action)
        return state.reshape(1, -1), reward, done, info

    def reset(self):
        init_state = super().reset()
        return init_state.reshape(1, -1)

def load_tutorbot_dataset(df: pd.DataFrame, env: TutorBot,
                        prep_for_is_ope=False) -> ProbabilityMDPDataset:
    """
    Can consider merging this w/ Sepsis one since it's largely identical

    :param df:
    :param env:
    :param prep_for_is_ope:
    :return:
    """
    features_list = []
    actions_list = []
    actions_prob_list = []
    rewards_list = []
    terminals_list = []

    for group_name, df_group in df.groupby(env.get_trajectory_marking_name()):
        features = df_group[env.get_feature_names()].to_numpy().astype(float)
        actions_prob = df_group[env.get_action_names()].to_numpy().astype(float)
        actions = df_group[env.get_action_names()].to_numpy().argmax(axis=1).astype(float)
        rewards = df_group[env.get_reward_name()].to_numpy().astype(float)
        terminals = np.array((rewards.shape[0] - 1) * [0.] + [1.]).astype(float)  # 1 is terminal state

        features_list.append(features)
        actions_list.append(actions)
        actions_prob_list.append(actions_prob)
        rewards_list.append(rewards)
        terminals_list.append(terminals)

    features = np.concatenate(features_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    actions_prob = np.concatenate(actions_prob_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    terminals = np.concatenate(terminals_list, axis=0)

    if not prep_for_is_ope:
        dataset = ProbabilityMDPDataset(
            observations=features,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            discrete_action=True
        )
        dataset.observed_actions = actions
        dataset.action_probabilities = actions_prob
        dataset.action_as_probability = False
        dataset.is_observed_action_discrete = True
    else:
        dataset = ProbabilityMDPDataset(
            observations=features,
            actions=actions_prob,
            rewards=rewards,
            terminals=terminals,
            discrete_action=False
        )
        dataset.observed_actions = actions
        dataset.action_probabilities = actions_prob
        dataset.action_as_probability = True
        dataset.is_observed_action_discrete = True

    return dataset

def evaluate_on_tutorbot_environment(
        env: TutorBot, n_trials: int = 10, epsilon: float = 0.0, batch_size: int = 64
) -> Callable[..., float]:
    """
    We iterate through all the possible states in the MDP.
    :param env:
    :param n_trials:
    :param epsilon:
    :param batch_size: for all_states
    :return:
    """

    def scorer(algo: AlgoProtocol, *args: Any) -> float:

        np.random.seed(1234)
        random.seed(1234)
        n_trials = 1000

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict(observation)[0]
                observation, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)

        return float(np.mean(episode_rewards))

    return scorer
