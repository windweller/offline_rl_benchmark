"""
We define a dataset protocol here that is used to require MDPDataset to have
some properties.
"""

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from typing_extensions import Protocol

# TODO: maybe make this into a class, similar to the QlearningWrapper to make typechecking work...
class ProbabilityMDPDatasetProtocol(Protocol):
    """
    Once we augment MDPDataset with these 3 fields
    we can switch between discrete actions and action probabilities
    """
    actions: np.ndarray
    observations: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray

    action_probabilities: np.ndarray
    action_as_probability: bool  # whether the actions array has probability for each action

#     @property
#     def actions(self) -> np.ndarray:
#         return self._actions
#
#     @actions.setter
#     def actions(self, actions: np.ndarray) -> None:
#         self._actions = actions
#
#     @property
#     def action_probabilities(self) -> np.ndarray:
#         return self._action_probabilities
#
#     @action_probabilities.setter
#     def action_probabilities(self, action_probabilities: np.ndarray) -> None:
#         self._action_probabilities = action_probabilities
#
#     @property
#     def action_as_probability(self) -> bool:
#         return self._action_as_probability
#
#     @action_as_probability.setter
#     def action_as_probability(self, action_as_probability: bool) -> None:
#         self._action_as_probability = action_as_probability