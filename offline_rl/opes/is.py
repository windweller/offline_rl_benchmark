"""
IS is more of a scorer, because it's non-parametric.
We follow the d3rllpy.metrics.scorer format to define them
Such as:
average_value_estimation_scorer

However, this cannot be passed into the "metrics" argument of the fit() function
"""

from typing import List

import numpy as np
import torch

from d3rlpy.dataset import Episode, Transition
from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE
from offline_rl.algs.probabilistic_policy_wrappers import ProbabilisticPolicyProtocol

def check_if_action_is_proper_probability(transition: Transition):
    assert type(transition.action) != int, "In order to be evaluated, the action must be a probability"
    assert np.abs(np.sum(transition.action) - 1) < 1e-6, "In order to be evaluated, the action must be a probability"

def _wis_ope(pibs: torch.Tensor, pies: torch.Tensor, rewards: torch.Tensor, length: torch.Tensor,
             no_weight_norm: bool=False, clip_lower: float=1e-16, clip_upper: float=1e3):
    """
    Private function used by the scorer
    :return:
    """
    n = pibs.shape[0]
    max_time = int(torch.max(length))
    weights = torch.ones((n, max_time))

    for i in range(n):
        last = 1
        for t in range(int(length[i])):
            assert pibs[i, t] != 0
            last = last * (pies[i, t] / pibs[i, t])
            weights[i, t] = last
        weights[i, length[i]:] = weights[i, length[i] - 1]
    weights = torch.clip(weights, clip_lower, clip_upper)
    if not no_weight_norm:
        weights_norm = weights.sum(dim=0)
        weights /= weights_norm  # per-step weights (it's cumulative)
    else:
        weights /= n

    # return w_i associated with each N
    return weights[:, -1] * rewards.sum(dim=-1), weights[:, -1]

def importance_sampling_scorer(
    algo: ProbabilisticPolicyProtocol, episodes: List[Episode]
) -> float:
    """
    :param algo:
    :param episodes:
    :return:

    Plan: we use these algo to predict probabilities on the episdoes right?
    and then build pibs, pies, and then use the formula

    """
    check_if_action_is_proper_probability(episodes[0].transitions[0])
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions_probs = algo.predict_action_probabilities(batch.observations)

    raise NotImplementedError

def weighted_importance_sampling_scorer(
    algo: ProbabilisticPolicyProtocol, episodes: List[Episode]
) -> float:
    """
    :param algo:
    :param episodes:
    :return:
    """
    check_if_action_is_proper_probability(episodes[0].transitions[0])

    raise NotImplementedError

def per_decision_importance_sampling_scorer(
    algo: ProbabilisticPolicyProtocol, episodes: List[Episode]
) -> float:
    check_if_action_is_proper_probability(episodes[0].transitions[0])

    raise NotImplementedError

def consistently_weighted_per_decision_importance_sampling_scorer(
        algo: ProbabilisticPolicyProtocol, episodes: List[Episode]
) -> float:
    """
    :param algo:
    :param episodes:
    :return:
    """
    check_if_action_is_proper_probability(episodes[0].transitions[0])

    raise NotImplementedError