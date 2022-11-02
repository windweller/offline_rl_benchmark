from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from d3rlpy.algos import DQN, DiscreteCQL

from abc import ABC, abstractmethod

import gym
import numpy as np

class ProbabilisticActionWrapper(ABC):

    @abstractmethod
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class QLearningWrapper(object):
    def __init__(self, dqn: Union[DQN, DiscreteCQL]):
        """

        :param dqn:
        :param safety_threshold:
        :return:
        """
        self.dqn = dqn
        self.dqn_impl = dqn._impl

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"
        state = torch.from_numpy(state).float()
        if self.dqn._use_gpu:
            state = state.to(self.dqn._device)

        q_values = self.dqn_impl._q_func(state)  # (batch_size, n_actions)
        action_prob = F.softmax(q_values, dim=1)

        if self.dqn._use_gpu:
            action_prob = action_prob.cpu()

        return action_prob.detach().numpy()