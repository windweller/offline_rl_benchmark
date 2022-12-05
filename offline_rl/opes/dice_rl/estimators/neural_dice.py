import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, List

from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from offline_rl.opes.dice_rl.data.dataset import TransitionMiniBatchWithDiscount, TransitionWithDiscount
from offline_rl.envs.dataset import ProbabilityMDPDataset
from offline_rl.opes.dice_rl.networks.value_network import ValueNetwork

import torch.optim as optim
from offline_rl.algs.policy_evaluation_wrappers import DiscreteProbabilisticTorchPolicyProtocol, \
    ContinuousProbabilisticPolicyProtocol
from offline_rl.opes.dice_rl.data.dataset import InitialStepRandomIterator, StepNextStepRoundIterator, \
    InitialStepRoundIterator

from d3rlpy.torch_utility import TorchMiniBatch
from offline_rl.opes.dice_rl.networks.value_network import ValueNetwork, ValueNetworkWithAction
from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE

ProbabilisticPolicy = Union[
    DiscreteProbabilisticTorchPolicyProtocol, ContinuousProbabilisticPolicyProtocol]


class NeuralDice(object):
    def __init__(self,
                 nu_network: Union[ValueNetwork, ValueNetworkWithAction],
                 zeta_network: Union[ValueNetwork, ValueNetworkWithAction],
                 nu_learning_rate: float,
                 zeta_learning_rate: float,
                 # nu_optimizer: optim.Optimizer,
                 # zeta_optimizer: optim.Optimizer,
                 # lam_optimizer: AdamFactory,
                 gamma: float,
                 has_log_probability: bool,
                 categorical_action: bool,
                 zero_reward=False,
                 solve_for_state_action_ratio: bool = True,
                 f_exponent: float = 1.5,
                 primal_form: bool = False,
                 num_samples: Optional[int] = None,
                 primal_regularizer: float = 0.,
                 dual_regularizer: float = 1.,
                 norm_regularizer: bool = False,
                 nu_regularizer: float = 0.,
                 zeta_regularizer: float = 0.,
                 weight_by_gamma: bool = False,
                 clip_lower: float = 1e-3, clip_upper: float = 1e3,
                 grad_clip=1):
        """Initializes the solver.

            Args:
              nu_network: The nu-value network.
              zeta_network: The zeta-value network.
              nu_optimizer: The optimizer to use for nu.
              zeta_optimizer: The optimizer to use for zeta.
              lam_optimizer: The optimizer to use for lambda.
              gamma: The discount factor to use.
              has_log_probability: does the dataset have logged probability on behavior policy? (i.e., ProbabilityMDPDataset)
              categorical_action: does the dataset have discrete action or continuous action?
              zero_reward: Not including the reward in computing the residual.
              reward_fn: A function that takes in an EnvStep and returns the reward for
                that step. If not specified, defaults to just EnvStep.reward.
              solve_for_state_action_ratio: Whether to solve for state-action density
                ratio. Defaults to True.
              f_exponent: Exponent p to use for f(x) = |x|^p / p.
              primal_form: Whether to use primal form of DualDICE, which optimizes for
                nu independent of zeta. This form is biased in stochastic environments.
                Defaults to False, which uses the saddle-point formulation of DualDICE.
              num_samples: Number of samples to take from policy to estimate average
                next nu value. If actions are discrete, this defaults to computing
                average explicitly. If actions are not discrete, this defaults to using
                a single sample.
              primal_regularizer: Weight of primal varibale regularizer.
              dual_regularizer: Weight of dual varibale regularizer.
              norm_regularizer: Weight of normalization constraint.
              nu_regularizer: Regularization coefficient on nu network.
              zeta_regularizer: Regularization coefficient on zeta network.
              weight_by_gamma: Weight nu and zeta losses by gamma ** step_num.
        """
        self._nu_network = nu_network
        self._zeta_network = zeta_network
        self._zero_reward = zero_reward

        self._nu_optimizer = optim.Adam(nu_network.parameters(), lr=nu_learning_rate)
        self._zeta_optimizer = optim.Adam(zeta_network.parameters(), lr=zeta_learning_rate)

        self._nu_regularizer = nu_regularizer
        self._zeta_regularizer = zeta_regularizer
        self._weight_by_gamma = weight_by_gamma

        self._gamma = gamma
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.grad_clip = grad_clip

        self._categorical_action = categorical_action
        self._solve_for_state_action_ratio = solve_for_state_action_ratio
        if (not self._solve_for_state_action_ratio and
                not has_log_probability):
            raise ValueError('Dataset must contain log-probability when '
                             'solve_for_state_action_ratio is False.')

        self._num_samples = num_samples  # for continuous action...
        if not self._categorical_action and self._num_samples is None:
            self._num_samples = 1

        self._primal_form = primal_form
        self._primal_regularizer = primal_regularizer
        self._dual_regularizer = dual_regularizer
        self._norm_regularizer = norm_regularizer
        self._lam = torch.tensor([0.], requires_grad=True)

        self._lam_optimizer = optim.Adam([self._lam], lr=nu_learning_rate)

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x) ** f_exponent / f_exponent
        self._fstar_fn = lambda x: torch.abs(x) ** fstar_exponent / fstar_exponent

    def tensor_to_gpu(self, batch: Union[TransitionMiniBatchWithDiscount,
                   TransitionMiniBatch], device: str) -> TorchMiniBatch:
        if type(batch) == TransitionMiniBatchWithDiscount:
            batch: TransitionMiniBatchWithDiscount
            new_batch = TorchMiniBatch(batch.transitions, device=device)  # policy.device()
        else:
            new_batch = TorchMiniBatch(batch, device=device)
        return new_batch

    def np_to_th(self, array: np.ndarray, device: str) -> torch.Tensor:
        return torch.from_numpy(array).to(device)

    def _get_value(self, network: Union[ValueNetwork, ValueNetworkWithAction],
                   env_step: Union[TransitionMiniBatchWithDiscount, TransitionMiniBatch], device: str) -> torch.Tensor:
        env_step = self.tensor_to_gpu(env_step, device)
        if self._solve_for_state_action_ratio:
            # shape matches, ok
            return network(env_step.observations, env_step.actions)
        else:
            return network(env_step.observations)

    def _get_average_value(self, network: Union[ValueNetwork, ValueNetworkWithAction],
                           env_step: Union[TransitionMiniBatchWithDiscount, TransitionMiniBatch],
                           policy: Union[
                DiscreteProbabilisticTorchPolicyProtocol, ContinuousProbabilisticPolicyProtocol]):

        env_step = self.tensor_to_gpu(env_step, policy.device())
        if self._solve_for_state_action_ratio:
            # discrete environment
            if self._categorical_action and self._num_samples is None:
                policy: DiscreteProbabilisticTorchPolicyProtocol
                # (batch_size, action_dim)
                actions_probs = policy.predict_action_probabilities(env_step.observations)
                batch_size, num_actions = actions_probs.shape[0], actions_probs.shape[1]
                actions = torch.ones(batch_size, 1) * torch.arange(num_actions)[None, :]
            else:
                # here we need to sample...
                # TODO: will do this...for continuous action
                raise NotImplementedError

            flat_actions = torch.reshape(actions, [batch_size * num_actions] +
                                         list(actions.shape[2:]))
            # create tiles of observations
            flat_observations = torch.reshape(
                torch.tile(env_step.observations[:, None, :], [1, num_actions] +
                           [1] * len(env_step.observations.shape[1:])),
                [batch_size * num_actions] + list(env_step.observations.shape[1:]))

            # predicted density ratios??
            # No, this is still just predicting values...hmmm
            # how the hell is this "solving for state-action density ratio"?
            # TODO: I don't know why the shape change is here...
            flat_values = network(flat_observations, flat_actions)
            values = torch.reshape(flat_values, [batch_size, num_actions])
            # + list(flat_values.shape[1:]

            return torch.mean(values * actions_probs, dim=1, keepdim=True)
        else:
            return network(env_step.observations)

    def train_loss(self, initial_env_step: TransitionMiniBatchWithDiscount,
                   env_step: TransitionMiniBatchWithDiscount, next_env_step: TransitionMiniBatchWithDiscount,
                   target_policy: Union[
                       DiscreteProbabilisticTorchPolicyProtocol, ContinuousProbabilisticPolicyProtocol]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The dataset must be MDPProbabilityDataset if solve_for_state_action_ratio is True.
        :return:
        """
        nu_values = self._get_value(self._nu_network, env_step, target_policy.device())
        initial_nu_values = self._get_average_value(self._nu_network,
                                                    initial_env_step, target_policy)
        next_nu_values = self._get_average_value(self._nu_network, next_env_step,
                                                 target_policy)

        zeta_values = self._get_value(self._zeta_network, env_step, target_policy.device())

        # numpy array still, need to lift to tensor...
        discounts = torch.from_numpy(self._gamma * next_env_step.discounts).to(target_policy.device())  # [batch_size]
        policy_ratio = 1.0
        batch = self.tensor_to_gpu(env_step, target_policy.device())
        if not self._solve_for_state_action_ratio:
            # get log probability of policy, and logged probability from the dataset
            # batch = self.tensor_to_gpu(env_step, target_policy.device())
            policy_probabilities = target_policy.predict_action_probabilities(batch.observations)
            logged_probabilities = batch.actions
            policy_ratio = policy_probabilities / logged_probabilities
            policy_ratio = torch.clip(policy_ratio, self.clip_lower, self.clip_upper)

        bellman_residuals = (discounts * policy_ratio).reshape(-1, 1) * next_nu_values - nu_values \
                            - self._norm_regularizer * self._lam
        if not self._zero_reward:
            bellman_residuals = bellman_residuals + policy_ratio * batch.rewards

        zeta_loss = -zeta_values * bellman_residuals
        nu_loss = (1 - self._gamma) * initial_nu_values
        lam_loss = self._norm_regularizer * self._lam
        if self._primal_form:
            nu_loss = nu_loss + self._fstar_fn(bellman_residuals)
            lam_loss = lam_loss + self._fstar_fn(bellman_residuals)
        else:
            nu_loss = nu_loss + zeta_values * bellman_residuals
            lam_loss = lam_loss - self._norm_regularizer * zeta_values * self._lam

        nu_loss = nu_loss + self._primal_regularizer * self._f_fn(nu_values)
        zeta_loss = zeta_loss + self._dual_regularizer * self._f_fn(zeta_values)

        if self._weight_by_gamma:
            # TODO: I'm not sure why they do this line:
            # weights = self._gamma**tf.cast(env_step.step_num, tf.float32)[:, None]
            weights = torch.from_numpy(env_step.discounts).to(target_policy.device())
            weights /= 1e-6 + torch.mean(discounts)
            nu_loss = nu_loss * weights
            zeta_loss = zeta_loss * weights

        return nu_loss, zeta_loss, lam_loss

    def _orthogonal_regularization(self, encoder: Union[ValueNetwork, ValueNetworkWithAction]):
        reg = 0
        for layer in encoder._layers:
            if type(layer) == nn.Linear:
                prod = torch.matmul(torch.transpose(layer.weight, 0, 1), layer.weight)
                reg = reg + torch.sum(torch.square(prod * (1 - torch.eye(prod.shape[0]))))
        return reg

    def train_step(self, initial_env_step: TransitionMiniBatchWithDiscount,
                   env_step: TransitionMiniBatchWithDiscount, next_env_step: TransitionMiniBatchWithDiscount,
                   target_policy: Union[
                       DiscreteProbabilisticTorchPolicyProtocol, ContinuousProbabilisticPolicyProtocol]):
        nu_loss, zeta_loss, lam_loss = self.train_loss(initial_env_step, env_step,
                                                       next_env_step,
                                                       target_policy)
        nu_loss = nu_loss + self._nu_regularizer * self._orthogonal_regularization(
            self._nu_network)
        zeta_loss = zeta_loss + self._zeta_regularizer * self._orthogonal_regularization(
            self._zeta_network)

        with torch.autograd.set_detect_anomaly(True):
            self._nu_optimizer.zero_grad()
            self._zeta_optimizer.zero_grad()
            self._lam_optimizer.zero_grad()

            nu_loss.mean().backward(retain_graph=True, inputs=list(self._nu_network.parameters()))
            torch.nn.utils.clip_grad_norm_(self._nu_network.parameters(), self.grad_clip)
            self._nu_optimizer.step()

            zeta_loss.mean().backward(retain_graph=True, inputs=list(self._zeta_network.parameters()))
            torch.nn.utils.clip_grad_norm_(self._zeta_network.parameters(), self.grad_clip)
            self._zeta_optimizer.step()

            lam_loss.mean().backward(inputs=[self._lam])
            torch.nn.utils.clip_grad_norm_(self._lam, self.grad_clip)
            self._lam_optimizer.step()

        return (torch.mean(nu_loss), torch.mean(zeta_loss), torch.mean(lam_loss))

    def get_fullbatch_average(self, dataset: ProbabilityMDPDataset, device: str):
        # we should write this like a scorer
        total_values = []
        episodes = dataset.episodes
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, 1):
                zeta = self._get_value(self._zeta_network, batch, device)
                rewards = torch.from_numpy(batch.rewards).to(device)
                weighted_reward = (zeta * rewards).sum().cpu().detach().numpy()
                total_values.append(weighted_reward)

        return np.mean(total_values)

    def get_fullbatch_average_for_nu_zero(self, dataset: ProbabilityMDPDataset):
        raise Exception("This is not needed for getting average reward estimate, we will implement this later...")

    def estimate_average_reward(self, dataset: ProbabilityMDPDataset,
                                target_policy: ProbabilisticPolicy):
        # actually quite simple, just need to get zeta (weight)
        # and just compute E[zeta * reward]
        def weight_fn(env_step: TransitionMiniBatchWithDiscount):
            zeta = self._get_value(self._zeta_network, env_step, target_policy.device())
            policy_ratio = 1.0
            if not self._solve_for_state_action_ratio:
                # get log probability of policy, and logged probability from the dataset
                batch = self.tensor_to_gpu(env_step, target_policy.device())
                policy_probabilities = target_policy.predict_action_probabilities(batch.observations)
                logged_probabilities = batch.actions
                policy_ratio = policy_probabilities / logged_probabilities
                policy_ratio = torch.clip(policy_ratio, self.clip_lower, self.clip_upper)

            return zeta * policy_ratio

        def init_nu_fn(initial_env_step: TransitionMiniBatchWithDiscount):
            """Computes average initial nu values of episodes."""
            # env_step is an episode, and we just want the first step.
            value = self._get_average_value(self._nu_network, initial_env_step,
                                            target_policy)
            return value

        dual_step = self.get_fullbatch_average(dataset, target_policy.device())

        return dual_step