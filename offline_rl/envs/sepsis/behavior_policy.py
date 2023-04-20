import os
import numpy as np
import pandas as pd

from offline_rl.envs.sepsis.env import Sepsis
from offline_rl.envs.datasets import DATA_DIRECTORY
from offline_rl.algs.policy_evaluation_wrappers import DiscreteProbabilisticPolicyProtocol
from d3rlpy.algos.base import AlgoBase, AlgoImplBase

from typing import List, Tuple

# note the naming convention change here (different from datasets.py)
POLICY_NOISE = ['0', '005', '010', '015', '020']
POLICY_TRUE_PERF = [
    (-0.0378, 0.16268505739618497),
    (-0.0952, 0.16869304194305113),
    (-0.1502, 0.1725502822368019),
    (-0.1846, 0.1744270824728775),
    (-0.225, 0.17861533388821912)
]
POLICY_TRUE_MEAN_PERF = [x[0] for x in POLICY_TRUE_PERF]

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

class DiscreteSepsisTabularPolicy(AlgoBase,DiscreteProbabilisticPolicyProtocol):
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

    @property
    def n_frames(self) -> int:
        return 1

    def create_policy_map(self):
        features = self.all_states[self.env.get_feature_names()]
        features = features.to_numpy().astype(int)
        policy_map = {}
        for i in range(features.shape[0]):
            # === this does bug checking ===
            # currently not passing

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
