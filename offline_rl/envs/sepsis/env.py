"""
A Fake Gym Env for Sepsis
It will actually not run, but it will return the correct configuration
"""

import gym

import numpy as np
import offline_rl.envs.sepsis.counterfactual as cf
import pandas as pd
import pickle
import os

# Sepsis Simulator code
from offline_rl.envs.sepsis.State import State
from offline_rl.envs.sepsis.Action import Action
from offline_rl.envs.sepsis.DataGenerator import DataGenerator
from offline_rl.envs.dataset import DiscreteProbabilityMDPDataset, CSVDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from offline_rl.algs.discrete_policy_evaluation_wrappers import ProbabilisticPolicyProtocol

from d3rlpy.dataset import MDPDataset, Transition

class Sepsis(gym.Env, CSVDataset):
    """
    A Fake Gym Env for Sepsis
    It will actually not run, but it will return the correct configuration
    """

    NSIMSAMPS = 1000  # Samples to draw from the simulator
    NSTEPS = 20  # Max length of each trajectory
    NCFSAMPS = 5  # Counterfactual Samples per observed sample
    DISCOUNT_Pol = 0.99  # Used for computing optimal policies
    DISCOUNT = 1  # Used for computing actual reward
    PHYS_EPSILON = 0.05  # Used for sampling using physician pol as eps greedy

    PROB_DIAB = 0.2

    def __init__(self, env_name: str, all_states: pd.DataFrame, prep_for_is_ope: bool = False):
        """

        :param env_name: this is to determine mdp vs. pomdp, correct state size
        :param all_states: this is for test on the true environment
        :param prep_for_is_ope: this will set the correct action space, so that Alg.build_with_dataset and Alg.build_with_env
                                will work in the same way
        """
        self.env_name = env_name.split('-')[0]
        self.prep_for_is_ope = prep_for_is_ope
        self._init_env()

        self.all_states = all_states

    def _init_env(self):
        """
        Initialize the environment
        """
        self._init_state()
        self._init_action_space()
        self._init_observation_space()

    def _init_state(self):
        """
        Initialize the state
        """
        self.state = np.zeros(8)
        self.state_idx = 0

    def _init_action_space(self):
        """
        Initialize the action space
        """
        self.action_space = gym.spaces.Discrete(8)

    def _init_observation_space(self):
        """
        Initialize the observation space
        """
        if self.env_name == 'mdp':
            self.observation_space = gym.spaces.Box(low=np.array([0., 0, 0, 0, 0, 0, 0, 0]).astype('float32'),
                                                    high=np.array([3., 3, 5, 2, 2, 2, 2, 2]).astype('float32'))
        elif self.env_name == 'pomdp':
            self.observation_space = gym.spaces.Box(low=np.array([0., 0, 0, 0, 0, 0]).astype('float32'),
                                                    high=np.array([3., 3, 2, 2, 2, 2]).astype('float32'))
        else:
            raise Exception("Unknown environment name")

    def get_feature_names(self):
        if self.env_name == 'mdp':
            return ["hr_state", "sysbp_state", "glucose_state", "oxygen_state",
                      "diabetes_idx", "antibiotic_state", "vaso_state", "vent_state"]
        elif self.env_name == 'pomdp':
            return ['hr_state', 'sysbp_state', 'oxygen_state', 'antibiotic_state', 'vaso_state', 'vent_state']

    def get_action_names(self):
        # for data loading purposes
        return ["beh_p_0", "beh_p_1", "beh_p_2", "beh_p_3", "beh_p_4", "beh_p_5", "beh_p_6", "beh_p_7"]

    def get_reward_name(self):
        return "Reward"

    def get_trajectory_marking_name(self):
        return "Trajectory"

    def get_all_features(self):
        features = self.all_states[self.get_feature_names()]
        features = features.to_numpy().astype(float)
        return features

    def reset(self):
        """
        Reset the environment
        """
        self._init_state()
        return self.state

    def step(self, action):
        """
        Take a step in the environment
        """
        raise NotImplementedError

def load_sepsis_dataset(df: pd.DataFrame, env: Sepsis,
                        prep_for_is_ope=False) -> DiscreteProbabilityMDPDataset:
    """
    Load the sepsis dataset
    :param df: the dataframe to load
    :param prep_for_is_ope: whether to prepare the dataset for IS OPE. Since D3RLPy does not store action probabilities,
                            we need to directly add action probabilities to action (pretend it's a multi-dim continuous action space).
                            This will later be used by our OPE class to compute IS OPE.
                            However, if we flag is True and try to learn a discrete control policy (or FQE), it will fail.
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
        dataset = DiscreteProbabilityMDPDataset(
            observations=features,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            discrete_action=True
        )
        dataset.action_probabilities = actions_prob
        dataset.action_as_probability = False
    else:
        dataset = DiscreteProbabilityMDPDataset(
            observations=features,
            actions=actions_prob,
            rewards=rewards,
            terminals=terminals,
            discrete_action=False
        )
        dataset.action_as_probability = True

    return dataset

def evaluate_on_sepsis_environment(
        env: Sepsis, n_trials: int = 10, epsilon: float = 0.0, batch_size: int = 64
) -> Callable[..., float]:
    """
    We iterate through all the possible states in the MDP.
    :param env:
    :param n_trials:
    :param epsilon:
    :param batch_size: for all_states
    :return:
    """

    def scorer(algo: ProbabilisticPolicyProtocol, *args: Any) -> float:

        np.random.seed(1998)
        N_TRAJECTORIES = 5000
        dgen = DataGenerator()

        # we collect action distributions for each state
        action_distributions = []
        all_feats = env.get_all_features()
        num_batches = all_feats.shape[0] // batch_size

        for state in np.array_split(all_feats, num_batches, axis=0):

            action_distributions.append(algo.predict_action_probabilities(state)) # (batch_size, num_actions)

        policy = np.concatenate(action_distributions, axis=0) # (num_states, num_actions)

        states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals, difficulties = dgen.simulate(
            N_TRAJECTORIES, 20, policy=policy, policy_idx_type='full', output_state_idx_type='full',
            p_diabetes=env.PROB_DIAB, use_tqdm=False)

        std = rewards.flatten()
        std = np.std(std)

        mean_reward = sum(sum(rewards))[0]/ N_TRAJECTORIES

        return float(mean_reward)

    return scorer
