from typing import Any, ClassVar, Dict, List, Optional, Type, Tuple, cast

import gym
import numpy as np
import random
import torch

import d3rlpy
from d3rlpy.dataset import MDPDataset, Transition, Episode, TransitionMiniBatch
from d3rlpy.algos.utility import ModelBaseMixin
from d3rlpy.iterators import RoundIterator, TransitionIterator, RandomIterator

from offline_rl.data_augs.dynamics_runner import ModelRunner

DELETE = 'delete'
ADD = 'add'
MERGE = 'merge'


class Operation:
    operation_name: str

    def fit(self, transitions: List[Transition]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            transitions: list of transitions.

        """
        raise NotImplementedError

    def fit_with_dataset(self, dataset: MDPDataset) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    def transform(self, x: MDPDataset, ratio: float = 1.0) -> MDPDataset:
        """Returns processed observations.

        Args:
            x: observation.
            ratio: percentage of transitions transformed

        Returns:
            processed observation.

        """
        raise NotImplementedError


class MergeOperation(Operation):
    def operate(self, episodes: List[Episode]) -> Episode:
        raise NotImplementedError


class DeleteOperation(Operation):
    def operate(self, episode: Episode) -> Episode:
        raise NotImplementedError


# TODO: Potential ways:
#  1) replace with most similar s_t in the dataset
#  2) replace with s_t that has the most similar a_t in the dataset

class ReplaceOperation(Operation):
    model_runner: ModelRunner

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        rate: float = 1.0,
    ):
        self.rate = rate
        self.fit_with_dataset(dataset)

    def fit_with_dataset(self, dataset: MDPDataset) -> None:
        dynamics_encoder = d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=[200, 200, 200, 200],
            activation='swish',
        )

        dynamics_optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=2.5e-5)
        dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
            encoder_factory=dynamics_encoder,
            optim_factory=dynamics_optim,
            learning_rate=1e-3,
            n_ensembles=5,
            use_gpu=True,
        )

        dynamics_total_steps = 100000

        # train dynamics model
        dynamics.fit(dataset.episodes,
                     # eval_episodes=test_episodes,
                     n_steps=dynamics_total_steps,
                     save_interval=int(dynamics_total_steps / 10000),
                     # experiment_name=f"ProbabilisticEnsembleDynamics_{args.dataset}_{args.dynamics_total_steps}_{args.dynamics_lr}_{args.seed}"
                     )

        self.model_runner = ModelRunner(dynamics, dataset,
                                        batch_size=32,
                                        rollout_horizon=1)

    def operate(self, dataset: MDPDataset) -> MDPDataset:
        # MOPO sample more things
        # Since MOPO works, this should work too
        transitions: List[Transition] = []
        for ep in dataset.episodes:
            if random.uniform(0, 1) > self.rate:
                pass
            transitions.extend(ep.transitions)

        trajs = self.model_runner.generate_new_data(transitions)

        return trajs

# TODO: grab the nearest REAL state in the dataset after MOPO generate the new state
class ReplaceWithNearestOperation(Operation):
    def operate(self, transitions: List[Transition]) -> List[Transition]:
        raise NotImplementedError
