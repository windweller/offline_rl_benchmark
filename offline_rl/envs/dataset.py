from d3rlpy.dataset import MDPDataset
import numpy as np

class DiscreteProbabilityMDPDataset(MDPDataset):
    """
    This class functions the same as MDPDataset, but with 2 new fields
    It is a workaround for MDPDataset not being able to handle both discrete actions and action probabilities
    """
    action_probabilities: np.ndarray
    # whether the actions array has probability for each action; only for discrete actions
    action_as_probability: bool