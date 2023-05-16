from d3rlpy.dataset import MDPDataset, Transition, Episode
import numpy as np
from typing import List, Tuple, Iterator
import gym


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


class ProbabilityMDPDataset(MDPDataset):
    """
    This class functions the same as MDPDataset, but with 2 new fields
    It is a workaround for MDPDataset not being able to handle both discrete actions and action probabilities

    We only do this because we want to use dataset.episodes to get episodes and batches -- and they can only handle
    MDPDataset
    """
    action_probabilities: np.ndarray
    # whether the actions array has probability for each action; only for discrete actions
    action_as_probability: bool
    observed_actions: np.ndarray
    # because we need to flag is_discrete to store continuous probabilities
    is_observed_action_discrete: bool


def convert_dataset_for_is_ope(dataset: ProbabilityMDPDataset) -> ProbabilityMDPDataset:
    """
    Convert the dataset to be used for IS OPE

    Note that if action_as_probability is True, it means that this dataset currently is used for training
    We can convert it for IS OPE

    :param dataset:
    :return:
    """
    assert dataset.action_as_probability is False, "The dataset is in the correct format"
    new_dataset = ProbabilityMDPDataset(
        observations=dataset.observations,
        actions=dataset.action_probabilities,
        rewards=dataset.rewards,
        terminals=dataset.terminals,
        discrete_action=False  # needs to be false to keep all probability
    )
    new_dataset.action_probabilities = dataset.action_probabilities
    new_dataset.action_as_probability = True
    new_dataset.observed_actions = dataset.observed_actions
    new_dataset.is_observed_action_discrete = dataset.is_observed_action_discrete
    return new_dataset


def convert_is_ope_dataset_for_training(dataset: ProbabilityMDPDataset) -> ProbabilityMDPDataset:
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

    new_dataset = ProbabilityMDPDataset(
        observations=dataset.observations,
        actions=dataset.action_probabilities.argmax(axis=1),
        rewards=dataset.rewards,
        terminals=dataset.terminals,
        discrete_action=dataset.is_observed_action_discrete
    )
    new_dataset.action_probabilities = dataset.action_probabilities
    new_dataset.action_as_probability = True
    new_dataset.observed_actions = dataset.observed_actions
    new_dataset.is_observed_action_discrete = dataset.is_observed_action_discrete

    return new_dataset


class TransitionMiniBatch:
    _transitions: List[Transition]
    _observations: np.ndarray
    _actions: np.ndarray
    _rewards: np.ndarray
    _next_observations: np.ndarray
    _terminals: np.ndarray
    _n_steps: np.ndarray

    def __init__(self, transitions: List[Transition], n_frames: int, n_steps=1):
        self._transitions = transitions

        self._transitions = transitions
        observation_shape = transitions[0].get_observation_shape()
        observation_ndim = len(observation_shape)
        observation_dtype = transitions[0].observation.dtype

        # determine action shape
        action_size = transitions[0].get_action_size()
        action_shape = tuple()
        action_dtype = np.int32
        if isinstance(transitions[0].action, np.ndarray):
            action_shape = (action_size,)
            action_dtype = np.float32

        size = len(transitions)

        self._observations = np.empty(
            (size,) + observation_shape, dtype=observation_dtype
        )
        self._actions = np.empty((size,) + action_shape, dtype=action_dtype)
        self._rewards = np.empty((size, 1), dtype=np.float32)
        self._next_observations = np.empty(
            (size,) + observation_shape, dtype=observation_dtype
        )
        self._terminals = np.empty((size, 1), dtype=np.float32)
        self._n_steps = np.empty((size, 1), dtype=np.float32)

        for i, transition in enumerate(transitions):
            self._observations[i] = transition.observation
            self._actions[i] = transition.action
            self._rewards[i] = transition.reward
            self._next_observations[i] = transition.next_observation
            self._terminals[i] = transition.terminal
            self._n_steps[i] = 1

    @property
    def observations(self):
        """ Returns mini-batch of observations at `t`.
        Returns:
            numpy.ndarray or torch.Tensor: observations at `t`.
        """
        return self._observations

    @property
    def actions(self):
        """ Returns mini-batch of actions at `t`.
        Returns:
            numpy.ndarray: actions at `t`.
        """
        return self._actions

    @property
    def rewards(self):
        """ Returns mini-batch of rewards at `t`.
        Returns:
            numpy.ndarray: rewards at `t`.
        """
        return self._rewards

    @property
    def next_observations(self):
        """ Returns mini-batch of observations at `t+n`.
        Returns:
            numpy.ndarray or torch.Tensor: observations at `t+n`.
        """
        return self._next_observations

    @property
    def terminals(self):
        """ Returns mini-batch of terminal flags at `t+n`.
        Returns:
            numpy.ndarray: terminal flags at `t+n`.
        """
        return self._terminals

    @property
    def n_steps(self):
        """ Returns mini-batch of the number of steps before next observations.
        This will always include only ones if ``n_steps=1``. If ``n_steps`` is
        bigger than ``1``. the values will depend on its episode length.
        Returns:
            numpy.ndarray: the number of steps before next observations.
        """
        return self._n_steps

    @property
    def transitions(self):
        """ Returns transitions.
        Returns:
            d3rlpy.dataset.Transition: list of transitions.
        """
        return self._transitions

    def size(self):
        """ Returns size of mini-batch.
        Returns:
            int: mini-batch size.
        """
        return len(self._transitions)

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self._transitions[index]

    def __iter__(self):
        return iter(self._transitions)


def _make_batches(
        episode: Episode, window_size: int, n_frames: int
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions, n_frames)
        yield batch


# ===== The following two functions work with any D3rlpy MDPDataset =====
# They have been tested to be able to recreate the same dataset
# And is generally efficient

def episodes_to_MDPDataset(episodes: List[Episode], env: gym.Env) -> MDPDataset:
    # first iteration to determine shape

    size = sum(ep.size() + 1 for ep in episodes)
    observation_dim = env.observation_space.shape[0]
    # this only works with discrete action space
    # action_dim = env.action_space.n

    observations = np.zeros((size, observation_dim))
    actions = np.zeros((size))
    rewards = np.zeros(size)
    terminals = np.zeros(size)
    episode_terminals = np.zeros(size)

    prev_i = 0
    for ep in episodes:
        actual_size = ep.observations.shape[0]
        observations[prev_i:prev_i + ep.observations.shape[0]] = ep.observations
        actions[prev_i:prev_i + ep.rewards.shape[0]] = ep.actions
        rewards[prev_i:prev_i + ep.rewards.shape[0]] = ep.rewards
        ep_terminals = np.zeros(ep.rewards.shape[0])
        ep_terminals[-1] = 1.
        episode_terminals[prev_i:prev_i + ep.rewards.shape[0]] = ep_terminals

        ep_terminals = np.zeros(actual_size)
        ep_terminals[-1] = int(ep.terminal)
        terminals[prev_i:prev_i + ep.rewards.shape[0]] = ep_terminals
        prev_i += actual_size

    return MDPDataset(observations=observations,
                      actions=actions,
                      rewards=rewards,
                      terminals=terminals,
                      episode_terminals=episode_terminals)


def sample_bootstrap(episodes: List[Episode], env: gym.Env,
                     n_copies=100, k_prop=0.2) -> Tuple[List[MDPDataset], float]:
    """
    We do n/k sample, where k < n
    But in argument, we set a proportion: k_prop so that k is dynamic

    :param episodes:
    :param env:
    :param n_copies:
    :param k_prop:
    :return: a list of MDPDataset, and the ratio of n/k
    """

    datasets = []
    # traj_dist = np.zeros(len(episodes))  # this is for efficient bootstrap
    k = int(len(episodes) * k_prop)
    indices = np.arange(len(episodes))
    for i in range(n_copies):
        # sample with replacement
        sampled_indices = np.random.choice(indices, k, replace=True)
        sampled_episodes = [episodes[i] for i in sampled_indices]
        dataset = episodes_to_MDPDataset(sampled_episodes, env)
        datasets.append(dataset)

    return datasets, k / len(episodes)
