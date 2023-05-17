"""
IS is more of a scorer, because it's non-parametric.
We follow the d3rllpy.metrics.scorer format to define them
Such as:
average_value_estimation_scorer

However, this cannot be passed into the "metrics" argument of the fit() function
"""

from typing import List, Tuple

import numpy as np
import torch

from d3rlpy.dataset import Episode, Transition
from d3rlpy.metrics.scorer import WINDOW_SIZE
from offline_rl.envs.dataset import _make_batches
from offline_rl.algs.policy_evaluation_wrappers import DiscreteProbabilisticPolicyProtocol


def check_if_action_is_proper_probability(transition: Transition):
    assert type(
        transition.action) != np.int32, "In order to be evaluated, the action must be a probability, try convert_dataset_for_is_ope()"
    assert np.abs(np.sum(
        transition.action) - 1) < 1e-6, "In order to be evaluated, the action must be a probability, try convert_dataset_for_is_ope()"


def _wis_ope(pibs: np.ndarray, pies: np.ndarray, rewards: np.ndarray, length: np.ndarray, max_time: int,
             no_weight_norm: bool = False, clip_lower: float = 1e-16, clip_upper: float = 1e3,
             no_clip=False) -> Tuple[float, np.ndarray]:
    """
    Private function used by the scorer
    :return:
    """
    n = pibs.shape[0]
    weights = np.ones((n, max_time))

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0, print(pibs[i])
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last
        weights[i, int(length[i]):] = weights[i, int(length[i]) - 1]
    if not no_clip:
        weights = np.clip(weights, clip_lower, clip_upper)
    if not no_weight_norm:
        weights_norm = weights.sum(axis=0)
        weights /= weights_norm  # per-step weights (it's cumulative)
    else:
        weights /= n

    # return w_i associated with each N
    return (weights[:, -1] * rewards.sum(axis=-1)).sum(axis=0), weights[:, -1]


def _cwpdis_ope(pibs: np.ndarray, pies: np.ndarray, rewards: np.ndarray, length: np.ndarray, max_time: int,
                no_weight_norm: bool = False,
                clip_lower: float = 1e-16, clip_upper: float = 1e3) -> float:
    n = pibs.shape[0]
    weights = np.ones((n, max_time))
    wis_weights = np.ones(n)

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last

        wis_weights[i] = last

    weights = np.clip(weights, clip_lower, clip_upper)
    if not no_weight_norm:
        weights_norm = weights.sum(axis=0)
        # step 1: \sum_n r_nt * w_nt
        weighted_r = (rewards * weights).sum(axis=0)
        # step 2: (\sum_n r_nt * w_nt) / \sum_n w_nt
        score = weighted_r / weights_norm
        # step 3: \sum_t ((\sum_n r_nt * w_nt) / \sum_n w_nt)
        score = score.sum()
    else:
        score = (rewards * weights).sum(axis=1).mean()

    return score


def compute_pib_pie(algo: DiscreteProbabilisticPolicyProtocol, episodes: List[Episode]) -> Tuple[np.ndarray, np.ndarray,
                                                                                                 np.ndarray, np.ndarray,
                                                                                                 int]:
    n = len(episodes)
    max_t = max([len(episode) for episode in episodes])

    pibs, pies, rewards, lengths = np.zeros((n, max_t)), np.zeros((n, max_t)), np.zeros((n, max_t)), np.zeros((n))
    for idx, episode in enumerate(episodes):
        # this makes sure if an episode is too long, we chunk it
        all_actions_probs, all_behavior_action_probs, all_rewards = [], [], []
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions_probs = algo.predict_action_probabilities(batch.observations)
            all_actions_probs.append(actions_probs)
            all_behavior_action_probs.append(batch.actions)
            all_rewards.append(batch.rewards)

        # stack them, all are within an episode
        actions_probs = np.hstack(all_actions_probs)
        behavior_action_probs = np.hstack(all_behavior_action_probs)
        ep_rewards = np.hstack(all_rewards)

        T = len(episode)
        beh_probs = behavior_action_probs.max(axis=-1, initial=0)  # we take the highest probability (chosen action)
        pibs[idx, :T] = beh_probs

        chosen_actions = behavior_action_probs.argmax(axis=-1)
        pies[idx, :T] = [actions_probs[t, chosen_actions[t]] for t in range(T)]
        rewards[idx, :T] = ep_rewards.squeeze()

        lengths[idx] = T

    return pibs, pies, rewards, lengths, max_t


def importance_sampling_scorer(
        algo: DiscreteProbabilisticPolicyProtocol, episodes: List[Episode],
        clip_lower: float = 1e-16, clip_upper: float = 1e3, no_clip=False,
) -> float:
    """
    :param algo:
    :param episodes:
    :return:

    Plan: we use these algo to predict probabilities on the episdoes right?
    and then build pibs, pies, and then use the formula

    """
    check_if_action_is_proper_probability(episodes[0].transitions[0])
    pibs, pies, rewards, lengths, max_t = compute_pib_pie(algo, episodes)
    score, weights = _wis_ope(pibs, pies, rewards, lengths, max_t, no_weight_norm=True,
                              clip_lower=clip_lower, clip_upper=clip_upper, no_clip=no_clip)

    return score


def importance_sampling_scorer_with_weights(algo: DiscreteProbabilisticPolicyProtocol, episodes: List[Episode],
                                            clip_lower: float = 1e-16, clip_upper: float = 1e3, no_clip=False,
                                            ) -> Tuple[float, np.array]:
    check_if_action_is_proper_probability(episodes[0].transitions[0])
    pibs, pies, rewards, lengths, max_t = compute_pib_pie(algo, episodes)
    score, weights = _wis_ope(pibs, pies, rewards, lengths, max_t, no_weight_norm=True,
                              clip_lower=clip_lower, clip_upper=clip_upper, no_clip=no_clip)

    return score, weights


def wis_scorer(
        algo: DiscreteProbabilisticPolicyProtocol, episodes: List[Episode],
        clip_lower: float = 1e-16, clip_upper: float = 1e3, no_clip=False
) -> float:
    """
    :param algo:
    :param episodes:
    :return:
    """
    check_if_action_is_proper_probability(episodes[0].transitions[0])

    check_if_action_is_proper_probability(episodes[0].transitions[0])
    pibs, pies, rewards, lengths, max_t = compute_pib_pie(algo, episodes)
    score, weights = _wis_ope(pibs, pies, rewards, lengths, max_t, no_weight_norm=False,
                              clip_lower=clip_lower, clip_upper=clip_upper, no_clip=no_clip)

    return score


def pdis_scorer(
        algo: DiscreteProbabilisticPolicyProtocol, episodes: List[Episode],
        clip_lower: float = 1e-16, clip_upper: float = 1e3
) -> float:
    check_if_action_is_proper_probability(episodes[0].transitions[0])
    pibs, pies, rewards, lengths, max_t = compute_pib_pie(algo, episodes)
    score = _cwpdis_ope(pibs, pies, rewards, lengths, max_t, no_weight_norm=True,
                        clip_lower=clip_lower, clip_upper=clip_upper)
    return score


def cwpdis_scorer(
        algo: DiscreteProbabilisticPolicyProtocol, episodes: List[Episode],
        clip_lower: float = 1e-16, clip_upper: float = 1e3
) -> float:
    """
    :param algo:
    :param episodes:
    :return:
    """
    check_if_action_is_proper_probability(episodes[0].transitions[0])

    pibs, pies, rewards, lengths, max_t = compute_pib_pie(algo, episodes)
    score = _cwpdis_ope(pibs, pies, rewards, lengths, max_t, no_weight_norm=False,
                        clip_lower=clip_lower, clip_upper=clip_upper)
    return score
