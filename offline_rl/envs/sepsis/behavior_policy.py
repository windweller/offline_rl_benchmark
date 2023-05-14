import os
import numpy as np
import pandas as pd

from offline_rl.envs.sepsis.env import Sepsis
from offline_rl.envs.datasets import DATA_DIRECTORY
from offline_rl.algs.policy_evaluation_wrappers import DiscreteProbabilisticPolicyProtocol
from d3rlpy.algos.base import AlgoBase, AlgoImplBase

from typing import List, Tuple

# note the naming convention change here (different from datasets.py)
# POLICY_NOISE = ['0', '005', '010', '015', '020',
#                 '025', '030', '035', '040']

# this is the  performance on the original MDP
# however, we deal with a truncated MDP with H=20, so we will never get this performance
# POLICY_TRUE_PERF = [
#     (-0.0378, 0.16268505739618497),
#     (-0.0952, 0.16869304194305113),
#     (-0.1502, 0.1725502822368019),
#     (-0.1846, 0.1744270824728775),
#     (-0.225, 0.17861533388821912)
# ]

# so we should choose:
# 0, 0.1, 0.2, 0.3, 0.4
# to make it clearly distinguishable
# POLICY_TRUE_PERF = [
#     (-0.0018, 0.15879614899137934),  # 0
#     (-0.0045, 0.1646314567190225),  # 0.05
#     (-0.0072, 0.1725502822368019),  # 0.1
#     (-0.0088, 0.1744270824728775),  # 0.15
#     (-0.0107, 0.17861533388821912),  # 0.2
#     (-0.012695238095238095, 0.17593581624668006),  # 0.25
#     (-0.015819047619047618, 0.1782121881751605),  # 0.3
#     (-0.017619047619047618, 0.18140818669075603),  # 0.35
#     (-0.018095238095238095, 0.18209500937727374)  # 0.4
# ]

POLICY_NOISE = ['0', '010', '020', '030', '040']

POLICY_TRUE_PERF = [
    (-0.0018, 0.15879614899137934),  # 0
    (-0.0072, 0.1725502822368019),  # 0.1
    (-0.0107, 0.17861533388821912),  # 0.2
    (-0.0158, 0.1782121881751605),  # 0.3
    (-0.0181, 0.18209500937727374)  # 0.4
]

POLICY_TRUE_MEAN_PERF = [x[0] for x in POLICY_TRUE_PERF]

MDP_POLICY_TURE_PERF = [
    (0.019847619047619048, 0.15585896719920864),
    (0.011047619047619047, 0.1655930808013073),
    (0.003904761904761905, 0.17038969564941225),
    (-0.0017523809523809523, 0.17738065137948913),
    (-0.00760952380952381, 0.18039513595177578)
]


class DiscreteSepsisTabularPolicyImpl(AlgoImplBase):
    def __init__(self, policy, policy_map):
        self.policy = policy
        self.policy_map = policy_map

    def predict_best_action(
            self, x
    ) -> np.ndarray:
        probs = []
        for i in range(x.shape[0]):
            # we randomly assign probability
            if tuple(x[i, :].astype(int)) not in self.policy_map:
                probs.append(np.ones(8) / 8)
            else:
                probs.append(self.policy_map[tuple(x[i].astype(int))])
        return np.argmax(probs, axis=1)

    def save_policy(self, fname: str) -> None:
        pass

    def predict_value(
            self,
            x,
            action,
            with_std: bool):
        pass

    def sample_action(self, x) -> np.ndarray:
        pass

    def save_model(self, fname: str) -> None:
        pass

    def load_model(self, fname: str) -> None:
        pass

    @property
    def observation_shape(self):
        return 6

    @property
    def action_size(self) -> int:
        return 8


class DiscreteSepsisTabularPolicy(DiscreteProbabilisticPolicyProtocol):
    def __init__(self, env: Sepsis, policy_npz_path: str,
                 noise_level='05'):
        # load in the npy stuff
        self.noise_level: str = noise_level
        self.all_states: pd.DataFrame = env.all_states
        self.env = env

        self.poilcy_npz = np.load(policy_npz_path, allow_pickle=True)['policy']
        self.create_policy_map()

        # self.n_frames = 1
        self._impl = DiscreteSepsisTabularPolicyImpl(self, self.policy_map)

        # self._action_scaler = None
        # self._batch_size = 4
        # self._gamma = 0.99

    @property
    def n_frames(self) -> int:
        return 1

    def create_policy_map(self):
        features = self.all_states[self.env.get_feature_names()]
        features = features.to_numpy().astype(int)
        policy_map = {}
        for i in range(features.shape[0]):
            # === this does bug checking ===

            # if tuple(features[i]) in policy_map:
            #     assert np.sum(policy_map[tuple(features[i])] == self.poilcy_npz[i]) == 8, print(policy_map[tuple(features[i])],
            #                                                                            self.poilcy_npz[i])
            policy_map[tuple(features[i])] = self.poilcy_npz[i]
        self.policy_map = policy_map

    def predict_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        probs = []
        for i in range(state.shape[0]):
            # we randomly assign probability
            # if tuple(state[i, :].astype(int)) not in self.policy_map:
            #     print(state[i, :])
            #     probs.append(np.ones(8) / 8)
            # else:
            probs.append(self.policy_map[tuple(state[i].astype(int))])
        return np.array(probs)


def load_sepsis_ensemble_policies(env: Sepsis) -> List[DiscreteSepsisTabularPolicy]:
    policies = []
    for noise in POLICY_NOISE:
        filepath = f"{DATA_DIRECTORY}/ens_ope/marginalized_{noise}_policy.npz"
        if not os.path.exists(filepath):
            raise Exception("Must call get_sepsis_ensemble_datasets or other function to download data first")
        policy = DiscreteSepsisTabularPolicy(env, filepath, noise)
        policies.append(policy)
    return policies


def load_sepsis_ensemble_mdp_policies(env: Sepsis) -> List[DiscreteSepsisTabularPolicy]:
    policies = []
    for noise in POLICY_NOISE:
        filepath = f"{DATA_DIRECTORY}/ens_ope_gt/full_{noise}_policy.npz"
        if not os.path.exists(filepath):
            raise Exception("Must call get_sepsis_ensemble_datasets or other function to download data first")
        policy = DiscreteSepsisTabularPolicy(env, filepath, noise)
        policies.append(policy)
    return policies
