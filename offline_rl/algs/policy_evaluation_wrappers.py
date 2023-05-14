from typing_extensions import Any, Protocol

import torch
import torch.nn.functional as F
from d3rlpy.algos import DQN, DiscreteCQL
from d3rlpy.algos import BC, DiscreteBC
from d3rlpy.algos import BCQ, DiscreteBCQ
from d3rlpy.algos.torch.dqn_impl import DQNImpl
from d3rlpy.algos.torch.bc_impl import BCImpl, DiscreteBCImpl
from d3rlpy.algos.torch.bcq_impl import BCQImpl, DiscreteBCQImpl
from d3rlpy.algos.base import AlgoBase, AlgoImplBase

import gym
import numpy as np


class BasePolicyProtocol(Protocol):
    policy: AlgoBase
    policy_impl: AlgoImplBase

    def device(self) -> str:
        # we leave this for individual policy wrapper to implement
        raise NotImplementedError

    def use_gpu(self) -> bool:
        raise NotImplementedError

    @property
    def n_frames(self) -> int:
        return self.policy.n_frames

class DiscreteProbabilisticPolicyProtocol(BasePolicyProtocol):
    def predict_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DiscreteProbabilisticTorchPolicyProtocol(BasePolicyProtocol):
    def predict_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ContinuousProbabilisticPolicyProtocol(BasePolicyProtocol):
    def predict_action_probabilities(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DiscreteCQLEvaluationWrapper(DiscreteProbabilisticPolicyProtocol):
    r"""QLearningWrapper
    This wrapper handles state as numpy input.
    This should be used for final policy evaluation (with real environment)

    evaluation_policy = QLearningWrapper(discrete_cql)
    rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)

    """

    def __init__(self, dqn):
        """

        :param dqn:
        :param safety_threshold:
        :return:
        """
        self.policy = dqn
        self.policy_impl: DQNImpl = dqn._impl

    def device(self) -> str:
        return self.policy._use_gpu

    def use_gpu(self) -> bool:
        return bool(self.policy._use_gpu)

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


class DiscreteCQLTorchWrapper(DiscreteProbabilisticTorchPolicyProtocol):
    r"""QLearningWrapper
    This wrapper handles state as torch input.
    """

    def __init__(self, dqn):
        """

        :param dqn:
        :param safety_threshold:
        :return:
        """
        self.policy = dqn
        self.policy_impl: DQNImpl = dqn._impl

    def device(self) -> str:
        return self.policy._use_gpu

    def use_gpu(self) -> bool:
        return bool(self.policy._use_gpu)

    def predict_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"

        q_values = self.policy_impl._q_func(state)  # (batch_size, n_actions)
        action_prob = F.softmax(q_values, dim=1)

        return action_prob


class DiscreteBCEvaluationWrapper(DiscreteProbabilisticPolicyProtocol):
    r"""BC Wrapper
    """

    def __init__(self, bc: DiscreteBC):
        """

        :param bc:
        :return:
        """
        self.policy = bc
        self.policy_impl: DiscreteBCImpl = bc._impl

    def device(self) -> str:
        return self.policy._use_gpu

    def use_gpu(self) -> bool:
        return bool(self.policy._use_gpu)

    def predict_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"
        state = torch.from_numpy(state).float()

        if self.policy._use_gpu:
            state = state.to(self.policy._device)

        # Get action probability through BC or DiscreteBC
        action_prob = self.policy_impl._imitator(state).exp()  # (batch_size, n_actions)

        if self.policy._use_gpu:
            action_prob = action_prob.cpu()

        return action_prob.detach().numpy()

class DiscreteBCTorchWrapper(DiscreteProbabilisticTorchPolicyProtocol):
    r"""BC Wrapper
    """

    def __init__(self, bc: DiscreteBC):
        """

        :param bc:
        :return:
        """
        self.policy = bc
        self.policy_impl: DiscreteBCImpl = bc._impl

    def device(self) -> str:
        return self.policy._use_gpu

    def use_gpu(self) -> bool:
        return bool(self.policy._use_gpu)

    def predict_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"

        # Get action probability through BC or DiscreteBC
        action_prob = self.policy_impl._imitator(state).exp()  # (batch_size, n_actions)

        return action_prob

class BCEvaluationWrapper(ContinuousProbabilisticPolicyProtocol):
    def __init__(self, bc: BC):
        """

        :param bc:
        :return:
        """
        self.policy = bc
        self.policy_impl: BCImpl = bc._impl

    def device(self) -> str:
        return self.policy._device

    def use_gpu(self) -> bool:
        return self.policy._use_gpu

    def predict_action_probabilities(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()

        if self.policy._use_gpu:
            state = state.to(self.policy._device)
            action = action.to(self.policy._device)

        # Get action probability through BC
        normal_dist = self.policy_impl._imitator.dist(state)  # (batch_size, n_actions)
        action_prob = normal_dist.log_prob(action).exp()  # (batch_size, n_actions)

        if self.policy._use_gpu:
            action_prob = action_prob.cpu()

        return action_prob.detach().numpy()

class DiscreteBCQEvaluationWrapper(DiscreteProbabilisticPolicyProtocol):
    r"""BC Wrapper
    """

    def __init__(self, bc: DiscreteBCQ):
        """

        :param bc:
        :return:
        """
        self.policy = bc
        self.policy_impl: DiscreteBCQImpl = bc._impl

    def device(self) -> str:
        return self.policy._use_gpu

    def use_gpu(self) -> bool:
        return bool(self.policy._use_gpu)

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

class DiscreteBCQTorchWrapper(DiscreteProbabilisticTorchPolicyProtocol):
    r"""BC Wrapper
    """

    def __init__(self, bc: DiscreteBCQ):
        """

        :param bc:
        :return:
        """
        self.policy = bc
        self.policy_impl: DiscreteBCQImpl = bc._impl

    def device(self) -> str:
        return self.policy._use_gpu

    def use_gpu(self) -> bool:
        return bool(self.policy._use_gpu)

    def predict_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        assert len(state.shape) == 2, "cannot pass in a single state, needs to be batched"

        q_values = self.policy_impl._q_func(state)  # (batch_size, n_actions)
        action_prob = F.softmax(q_values, dim=1)

        return action_prob
