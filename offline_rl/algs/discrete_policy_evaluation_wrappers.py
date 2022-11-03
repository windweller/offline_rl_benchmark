from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast, Protocol

import torch
import torch.nn.functional as F
from d3rlpy.algos import DQN, DiscreteCQL
from d3rlpy.algos.torch.dqn_impl import DQNImpl
from d3rlpy.algos.base import AlgoBase, AlgoImplBase

import gym
import numpy as np


class BasePolicyProtocol(Protocol):
    policy: AlgoBase
    policy_impl: AlgoImplBase

    def device(self) -> torch.device:
        return self.policy.device()

    @property
    def n_frames(self) -> int:
        return self.policy.n_frames


class ProbabilisticPolicyProtocol(BasePolicyProtocol):
    def predict_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ProbabilisticTorchPolicyProtocol(BasePolicyProtocol):
    def predict_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QLearningEvaluationWrapper(ProbabilisticPolicyProtocol):
    r"""QLearningWrapper
    This wrapper handles state as numpy input.
    This should be used for final policy evaluation (with real environment)

    evaluation_policy = QLearningWrapper(cql)
    rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)

    """
    def __init__(self, dqn: Union[DQN, DiscreteCQL]):
        """

        :param dqn:
        :param safety_threshold:
        :return:
        """
        self.policy = dqn
        self.policy_impl: DQNImpl = dqn._impl

    def predict_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"
        state = torch.from_numpy(state).float()
        if self.policy._use_gpu:
            state = state.to(self.policy._device)

        q_values = self.policy_impl._q_func(state)  # (batch_size, n_actions)
        action_prob = F.softmax(q_values, dim=1)

        if self.policy._use_gpu:
            action_prob = action_prob.cpu()

        return action_prob.detach().numpy()

class QLearningTorchWrapper(ProbabilisticTorchPolicyProtocol):
    r"""QLearningWrapper
    This wrapper handles state as torch input.
    """
    def __init__(self, dqn: Union[DQN, DiscreteCQL]):
        """

        :param dqn:
        :param safety_threshold:
        :return:
        """
        self.policy = dqn
        self.policy_impl: DQNImpl = dqn._impl

    def predict_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"

        q_values = self.policy_impl._q_func(state)  # (batch_size, n_actions)
        action_prob = F.softmax(q_values, dim=1)

        return action_prob