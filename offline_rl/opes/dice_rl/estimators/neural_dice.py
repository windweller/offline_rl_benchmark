import torch
import numpy as np
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, List

from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from offline_rl.envs.dataset import ProbabilityMDPDataset
from offline_rl.opes.dice_rl.networks.value_network import ValueNetwork

import torch.optim as optim

def reverse_broadcast(input_tensor, target_tensor):
  input_rank = len(input_tensor.shape)
  target_rank = len(target_tensor.shape)
  additional_rank = max(0, target_rank - input_rank)
  return torch.reshape(input_tensor, list(input_tensor.shape) +
                    [1] * additional_rank)

class NeuralDice(object):
    """Policy evaluation with DICE."""

    def __init__(
            self,
            dataset_spec,
            env: ProbabilityMDPDataset,
            nu_network: ValueNetwork,
            zeta_network: ValueNetwork,
            nu_optimizer: optim.Optimizer,
            zeta_optimizer: optim.Optimizer,
            lam_optimizer: optim.Optimizer,
            gamma: Union[float, torch.Tensor],
            zero_reward: bool=False,
            reward_fn: Optional[Callable] = None,
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
    ):
        self._dataset_spec = dataset_spec
        self._nu_network = nu_network
        self._nu_network.create_variables()
        self._zeta_network = zeta_network
        self._zeta_network.create_variables()
        self._zero_reward = zero_reward

        self._nu_optimizer = nu_optimizer
        self._zeta_optimizer = zeta_optimizer
        self._lam_optimizer = lam_optimizer
        self._nu_regularizer = nu_regularizer
        self._zeta_regularizer = zeta_regularizer
        self._weight_by_gamma = weight_by_gamma

        self._gamma = gamma
        if reward_fn is None:
            # TODO: might not be correct for us
            reward_fn = lambda env_step: env_step.reward

        self._reward_fn = reward_fn
        self._num_samples = num_samples

        self._solve_for_state_action_ratio = solve_for_state_action_ratio
        # TODO: this is also not the dataset_spec we have...but might be close
        if (not self._solve_for_state_action_ratio and
                not self._dataset_spec.has_log_probability()):
            raise ValueError('Dataset must contain log-probability when '
                             'solve_for_state_action_ratio is False.')

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x) ** f_exponent / f_exponent
        self._fstar_fn = lambda x: torch.abs(x) ** fstar_exponent / fstar_exponent

        self._categorical_action = self._dataset_spec.is_categorical_action()

        if not self._categorical_action and self._num_samples is None:
            self._num_samples = 1

        self._primal_form = primal_form
        self._primal_regularizer = primal_regularizer
        self._dual_regularizer = dual_regularizer
        self._norm_regularizer = norm_regularizer
        self._lam = torch.Tensor([0.0])

    def _get_value(self, network, env_step: TransitionMiniBatch):
        # if we just do a 1:1 translation, it should be ok?
        if self._solve_for_state_action_ratio:
            return network((env_step.observations, env_step.actions))[0]
        else:
            return network(env_step.observations)[0]

    def _get_average_value(self, network, env_step: TransitionMiniBatch, policy):
        if self._solve_for_state_action_ratio:
            tfagents_step = env_step
            if self._categorical_action and self._num_samples is None:
                action_weights = policy.distribution(
                    tfagents_step).action.probs_parameter()
                action_dtype = self._dataset_spec.action.dtype
                batch_size = action_weights.shape[0]
                num_actions = action_weights.shape[-1]
                actions = torch.ones((batch_size, 1)) * torch.arange(num_actions, dtype=action_dtype)[None, :]
                # actions = (  # Broadcast actions
                #         tf.ones([batch_size, 1], dtype=action_dtype) *
                #         tf.range(num_actions, dtype=action_dtype)[None, :])
            else:
                raise NotImplementedError

                # batch_size = env_step.observation.shape[0]
                # num_actions = self._num_samples
                # action_weights = torch.ones([batch_size, num_actions]) / num_actions
                # actions = torch.stack(
                #     [policy.action(tfagents_step).action
                #      for _ in range(num_actions)],
                #     dim=1)

            flat_actions = torch.reshape(actions, [batch_size * num_actions] +
                                      list(actions.shape[2:]))
            flat_observations = torch.reshape(
                torch.tile(env_step.observations[:, None, ...],
                        [1, num_actions] + [1] * len(env_step.observations.shape[1:])),
                [batch_size * num_actions] + list(env_step.observations.shape[1:]))

            flat_values, _ = network((flat_observations, flat_actions))

            values = torch.reshape(flat_values, [batch_size, num_actions] +
                                list(flat_values.shape[1:]))
            return torch.sum(
                values * reverse_broadcast(action_weights, values), dim=1)
        else:
            return network(env_step.observations)[0]

    def _orthogonal_regularization(self, network):
        reg = 0
        for layer in network.layers:
            if isinstance(layer, torch.nn.Linear):
                prod = torch.matmul(torch.t(layer.weight), layer.weight)
                reg += torch.sum(torch.square(prod * (1 - torch.eye(prod.shape[0]))))
        return reg

    def train_loss(self, initial_env_step: TransitionMiniBatch, env_step: TransitionMiniBatch, next_env_step: TransitionMiniBatch,
                   policy):
        nu_values = self._get_value(self._nu_network, env_step)
        initial_nu_values = self._get_average_value(self._nu_network,
                                                    initial_env_step, policy)
        next_nu_values = self._get_average_value(self._nu_network, next_env_step, policy)

        zeta_values = self._get_value(self._zeta_network, env_step)

        discounts = self._gamma * next_env_step.gamma
        policy_ratio = 1.0
        if not self._solve_for_state_action_ratio:
            # tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
            policy_log_probabilities = policy.distribution(
                tfagents_step).action.log_prob(env_step.action)
            policy_ratio = tf.exp(policy_log_probabilities -
                                  env_step.get_log_probability())

        bellman_residuals = (
                reverse_broadcast(discounts * policy_ratio, nu_values) *
                next_nu_values - nu_values - self._norm_regularizer * self._lam)
        if not self._zero_reward:
            bellman_residuals += policy_ratio * self._reward_fn(env_step)

        zeta_loss = -zeta_values * bellman_residuals
        nu_loss = (1 - self._gamma) * initial_nu_values
        lam_loss = self._norm_regularizer * self._lam
        if self._primal_form:
            nu_loss += self._fstar_fn(bellman_residuals)
            lam_loss = lam_loss + self._fstar_fn(bellman_residuals)
        else:
            nu_loss += zeta_values * bellman_residuals
            lam_loss = lam_loss - self._norm_regularizer * zeta_values * self._lam

        nu_loss += self._primal_regularizer * self._f_fn(nu_values)
        zeta_loss += self._dual_regularizer * self._f_fn(zeta_values)

        if self._weight_by_gamma:
            weights = self._gamma ** tf.cast(env_step.step_num, tf.float32)[:, None]
            weights /= 1e-6 + torch.mean(weights)
            nu_loss *= weights
            zeta_loss *= weights

        return nu_loss, zeta_loss, lam_loss

