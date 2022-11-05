from d3rlpy.dataset import MDPDataset
import numpy as np

class CSVDataset(object):
    """
    When the offline RL dataset is in the CSV format
    We provide column names so that pandas library can better process
    """
    def get_feature_names(self):
        raise Exception

    def get_action_names(self):
        # for data loading purposes
        raise Exception

    def get_reward_name(self):
        raise Exception

    def get_trajectory_marking_name(self):
        raise Exception


class DiscreteProbabilityMDPDataset(MDPDataset):
    """
    This class functions the same as MDPDataset, but with 2 new fields
    It is a workaround for MDPDataset not being able to handle both discrete actions and action probabilities
    """
    action_probabilities: np.ndarray
    # whether the actions array has probability for each action; only for discrete actions
    action_as_probability: bool

def convert_dataset_for_is_ope(dataset: DiscreteProbabilityMDPDataset) -> DiscreteProbabilityMDPDataset:
    """
    Convert the dataset to be used for IS OPE

    Note that if action_as_probability is True, it means that this dataset currently is used for training
    We can convert it for IS OPE

    :param dataset:
    :return:
    """
    assert dataset.action_as_probability is False, "The dataset is in the correct format"
    dataset = DiscreteProbabilityMDPDataset(
        observations=dataset.observations,
        actions=dataset.action_probabilities,
        rewards=dataset.rewards,
        terminals=dataset.terminals,
        discrete_action=False
    )
    dataset.action_as_probability = True
    return dataset

def convert_is_ope_dataset_for_training(dataset: DiscreteProbabilityMDPDataset) -> DiscreteProbabilityMDPDataset:
    """
    Convert the dataset to be used for training

    Note that if action_as_probability is False, it means that this dataset currently is used for OPE
    We can convert it for Training

    :param dataset:
    :return:
    """
    assert dataset.action_as_probability is True, "The dataset is already in the correct format"
    # step 1: copy actions (action probabilities) to action_probabilities
    # step 2: override actions to discrete actions
    # step 3: set the flag

    new_dataset = DiscreteProbabilityMDPDataset(
        observations=dataset.observations,
        actions=dataset.action_probabilities.argmax(axis=1),
        rewards=dataset.rewards,
        terminals=dataset.terminals,
        discrete_action=False
    )
    new_dataset.action_probabilities = dataset.actions
    new_dataset.action_as_probability = False
    return new_dataset
