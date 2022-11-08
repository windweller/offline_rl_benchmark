"""
This file needs to be modular in a way
that we provide a class so that other methods
can use IS based OPE even for continuous domains.

https://github.com/LARG/regression-importance-sampling

Steps:
1. Train based on DiscreteBC (predict p(a|s) = 1)
2. Replace the action_probabilities in the dataset with the estimates (OR add to it if none exists)
3. Then we can plug into IS-based OPEs (directly without any change)
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from d3rlpy.algos import BC, DiscreteBC
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE

from offline_rl.envs.dataset import ProbabilityMDPDataset
from offline_rl.algs.discrete_policy_evaluation_wrappers import DiscreteBCEvaluationWrapper, BCEvaluationWrapper


class RegressionIS(object):
    """
    We either use Discrete BC (with one-hot, which is equivalent to log p)
    or we use Continuous BC (with stochastic policy to fit the Gaussian distribution)
    """

    def __init__(self, bc_estimator: Union[BC, DiscreteBC]) -> None:
        # verify that continuous BC uses stochastic policy
        self.bc_policy = bc_estimator
        if type(self.bc_policy) == BC:
            assert self.bc_policy.policy_type == 'stochastic'

        self.bc_estimator: Union[DiscreteBCEvaluationWrapper, BCEvaluationWrapper, None] = None

    def fit(self, dataset: MDPDataset, **kwargs: Any) -> None:
        # this dataset's action should either be continous or
        # discrete (with one-hot encoding of the actually observed action)

        # we need to check this dataset
        if dataset.is_action_discrete():
            assert type(dataset.episodes[0].transitions[0].action) == np.int32, "Should not pass in " \
                                                                           "DiscreteProbabilityMDPDataset, use original " \
                                                                           "MDPDataset for training instead."

        self.bc_policy.fit(dataset=dataset, **kwargs)
        if type(self.bc_policy) == DiscreteBC:
            self.bc_estimator = DiscreteBCEvaluationWrapper(self.bc_policy)
        else:
            self.bc_estimator = BCEvaluationWrapper(self.bc_policy)

    def estimate_bh_probabilities(self, dataset: MDPDataset) -> ProbabilityMDPDataset:
        """
        Estimate the behavior policy probabilities using the fitted BC estimator.
        Note, because the action probability in returned DiscreteProbabilityMDPDataset dataset
        is estimated, if you use `convert_dataset_for_is_ope`, `convert_is_ope_dataset_for_training`
        to convert the dataset, you will *NOT* get the original dataset back.

        :param dataset: MDPDataset
        :return:
        """
        assert self.bc_estimator is not None, "Please fit the estimator first."
        all_actions_probs = []
        for idx, episode in enumerate(dataset.episodes):
            for batch in _make_batches(episode, WINDOW_SIZE, self.bc_policy.n_frames):
                if type(self.bc_policy) == DiscreteBC:
                    # discrete BC
                    # (batch_size, n_actions)
                    action_probabilities = self.bc_estimator.predict_action_probabilities(batch.observations)
                else:
                    # continuous BC
                    # (batch_size, 1)
                    action_probabilities = self.bc_estimator.predict_action_probabilities(batch.observations,
                                                                                         batch.actions)
                all_actions_probs.append(action_probabilities)

        all_actions_probs = np.concatenate(all_actions_probs, axis=0)
        assert all_actions_probs.shape[0] == dataset.actions.shape[0], "The number of action probabilities should be the same as " \
                                                              "the number of transitions in the dataset."

        new_dataset = ProbabilityMDPDataset(
            observations=dataset.observations,
            actions=dataset.actions,
            rewards=dataset.rewards,
            terminals=dataset.terminals,
            discrete_action=dataset.is_action_discrete()
        )
        new_dataset.action_probabilities = all_actions_probs
        new_dataset.observed_actions = dataset.actions
        new_dataset.action_as_probability = False
        new_dataset.is_observed_action_discrete = dataset.is_action_discrete()

        return new_dataset
