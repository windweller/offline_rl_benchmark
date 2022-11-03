"""
We define a dataset protocol here that is used to require MDPDataset to have
some properties.
"""

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from typing_extensions import Protocol

class ProbabilityMDPDatasetProtocol(Protocol):
    """
    Once we augment MDPDataset with these 3 fields
    we can switch between discrete actions and action probabilities
    """
    actions: np.ndarray
    action_probabilities: np.ndarray
    action_as_probability: bool  # whether the actions array has probability for each action