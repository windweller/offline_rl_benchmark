from typing import Any, ClassVar, Dict, List, Optional, Type, Tuple, cast

import gym
import numpy as np
import torch

from d3rlpy.dataset import MDPDataset, Transition, Episode, TransitionMiniBatch
from d3rlpy.algos.utility import ModelBaseMixin
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from d3rlpy.iterators import RoundIterator, TransitionIterator, RandomIterator

class ModelRunner(ModelBaseMixin):
    def __init__(self, dynamics: ProbabilisticEnsembleDynamics, dataset: MDPDataset, batch_size: int,
                 rollout_horizon: int):
        self._dynamics = dynamics
        self.observation_shape = dynamics.observation_shape
        self.action_size = dynamics.action_size

        self._rollout_batch_size = batch_size
        self._rollout_horizon = rollout_horizon

        self.transitions = []
        self.init_transitions = []

        self.dataset = dataset

        for episode in dataset.episodes:
            self.transitions.extend(episode.transitions)
            self.init_transitions.append(episode.transitions[0])  # first transitions

        self.init_iterator = RoundIterator(self.init_transitions, batch_size)
        self.iterator = RoundIterator(self.transitions, batch_size)

    def sample_transitions(self):
        return self._sample_transitions()

    def sample_initial_transitions(self):
        return self._sample_initial_transitions(None)

    def _sample_transitions(
            self
    ) -> List[Transition]:
        # uniformly sample transitions
        n_transitions = self._rollout_batch_size
        assert n_transitions <= len(self.transitions), "batch size shouldn't be larger than what we can sample"

        indices = np.random.randint(len(self.transitions), size=n_transitions)
        return [self.transitions[i] for i in indices]

    def _sample_initial_transitions(
            self, transitions: List[Transition]
    ) -> List[Transition]:
        n_transitions = self._rollout_batch_size
        assert n_transitions <= len(self.init_transitions), "batch size shouldn't be larger than what we can sample"

        indices = np.random.randint(len(self.init_transitions), size=n_transitions)
        return [self.init_transitions[i] for i in indices]

    def _get_rollout_horizon(self):
        return self._rollout_horizon

    def _is_generating_new_data(self):
        return True

    def generate_next_state(
            self, observations: np.array
    ) -> Optional[np.array]:

        if not self._is_generating_new_data():
            return None

        # init_transitions = self._sample_initial_transitions(transitions)

        rets: List[Transition] = []

        # rollout
        # batch = TransitionMiniBatch(init_transitions)
        # observations = batch.observations
        actions = self._sample_rollout_action(observations)
        prev_transitions: List[Transition] = []
        for _ in range(1):
            # predict next state
            pred = self._dynamics.predict(observations, actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, rewards, variances = pred

            # regularize by uncertainty
            next_observations, rewards = self._mutate_transition(
                next_observations, rewards, variances
            )

            # sample policy action
            next_actions = self._sample_rollout_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(observations.shape[0]):
                transition = Transition(
                    observation_shape=self.observation_shape,
                    action_size=self.action_size,
                    observation=observations[i],
                    action=actions[i],
                    # next_action=actions[i],
                    # next_reward=float(rewards[i][0]),
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i].detach().numpy(),
                    terminal=0.0,
                    prev_transition=None,
                    next_transition=None
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.detach().numpy().copy()
            actions = next_actions.copy()

        return observations

    def generate_new_data(
        self, transitions: List[Transition]
    ) -> Optional[List[Transition]]:
        # assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        # assert self._dynamics, DYNAMICS_NOT_GIVEN_ERROR

        if not self._is_generating_new_data():
            return None

        init_transitions = self._sample_initial_transitions(transitions)

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = self._sample_rollout_action(observations)
        prev_transitions: List[Transition] = []
        for _ in range(self._get_rollout_horizon()):
            # predict next state
            pred = self._dynamics.predict(observations, actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, rewards, variances = pred

            # regularize by uncertainty
            next_observations, rewards = self._mutate_transition(
                next_observations, rewards, variances
            )

            # sample policy action
            next_actions = self._sample_rollout_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(len(init_transitions)):
                transition = Transition(
                    observation_shape=self.observation_shape,
                    action_size=self.action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i].detach().numpy(),
                    terminal=0.0,
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.detach().numpy().copy()
            actions = next_actions.copy()

        return rets