from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import numpy as np

from d3rlpy.ope.torch.fqe_impl import DiscreteFQEImpl, FQEBaseImpl
from offline_rl.envs.sepsis.behavior_policy import DiscreteSepsisTabularPolicy
from offline_rl.envs.sepsis.env import Sepsis

from tqdm import tqdm

"""
Currently it's fine that this only works with Sepsis

The goal:
1. load in the poilcy map (state map)
2. create a one-hot encoding
"""


class TabularFQE(object):
    n_frames: int = 1

    def __init__(self, algo: DiscreteSepsisTabularPolicy, lr=0.005, gamma=1.0):
        self.policy = algo
        self.lr = lr  # learning rate
        self.gamma = gamma

        self.Q = {}

    def build_with_env(self, sepsis: Sepsis):
        for tuple_state, _ in self.policy.policy_map.items():
            state_list = list(tuple_state)
            for i in range(sepsis.action_space.n):
                # (state, action)
                self.Q[tuple(state_list + [i])] = 0.

    def encode_state_action(self, state, action):
        state = state.astype(int)
        action = action.astype(int)

        return tuple(list(state) + [int(action)])

    def fit(self, test_episodes, n_epochs):
        for _ in range(n_epochs):
            for episode in test_episodes:
                for t, transition in enumerate(episode):
                    done = True if t == len(episode) - 1 else False
                    # step.observation, step.action, step.reward, step.next_observation
                    action = self.predict(np.array([transition.observation]))
                    s_a = self.encode_state_action(transition.observation, action)

                    # if done:
                    #     delta = 0
                    # else:
                    #     next_action = self.predict(np.array([transition.next_observation]))
                    #     next_s_a = self.encode_state_action(transition.next_observation, next_action)
                    #     delta = self.Q[next_s_a]
                    #
                    # self.Q[s_a] = self.Q[s_a] + self.lr * (transition.reward + self.gamma * delta - self.Q[s_a])

                    if done:
                        self.Q[s_a] = transition.reward
                    else:
                        next_action = self.predict(np.array([transition.next_observation]))
                        next_s_a = self.encode_state_action(transition.next_observation, next_action)
                        self.Q[s_a] = (1 - self.lr) * self.Q[s_a] + self.lr * (transition.reward + self.gamma * self.Q[next_s_a])


    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        # used by scorer, this is actually the evaluation policy's predicted action
        # I think this is a single value thingy
        if type(x) == list:
            x = np.array(x)
        # we have to sample here. This is because all our fixed sepsis policy is a noisy version of the optimal
        # but the noise isn't that high anyway to take out the optimal action
        # if we deterministically choose the max, then we will never see difference between policies

        # return self.policy._impl.predict_best_action(x)
        return self.policy._impl.sample_action(x)

    def predict_value(
            self,
            x: Union[np.ndarray, List[Any]],
            action: Union[np.ndarray, List[Any]],
            with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # used by scorer, this is the Q-value learned by FQE
        values = []
        if type(x) == list:
            x = np.array(x)
        for i in range(x.shape[0]):
            # a = action[i].astype(int)
            # s = x[i].astype(int)
            s_a = self.encode_state_action(x[i], action[i])
            if s_a not in self.Q:
                values.append(0.)
            else:
                values.append(self.Q[s_a])
            # scores.append()
        return np.array(values)
