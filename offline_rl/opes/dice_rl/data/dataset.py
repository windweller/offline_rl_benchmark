import abc
import collections
import numpy as np
import os
import os.path
import pickle

from typing import Any, Callable, Iterable, Tuple, Sequence, List, Union, cast, Iterator, NamedTuple

from collections import namedtuple
from d3rlpy.base import RoundIterator, RandomIterator
import torch

from abc import ABCMeta, abstractmethod

import offline_rl.opes.dice_rl.data.timestep as time_step

from offline_rl.envs.dataset import ProbabilityMDPDataset
from d3rlpy.dataset import TransitionMiniBatch, Transition
from d3rlpy.containers import FIFOQueue
from d3rlpy.iterators.base import TransitionIterator

TransitionWithDiscount = NamedTuple("TransitionWithDiscount",
                                    [("transition", Transition), ("discount", float)])
TransitionMiniBatchWithDiscount = NamedTuple("TransitionMiniBatchWithDiscount",
                                             [("transitions", TransitionMiniBatch), ("discounts", np.ndarray)])


class DiscountTransitionIterator(metaclass=ABCMeta):
    _transitions: List[TransitionWithDiscount]
    _generated_transitions: FIFOQueue[TransitionWithDiscount]
    _batch_size: int
    _n_steps: int
    _gamma: float
    _n_frames: int
    _real_ratio: float
    _real_batch_size: int
    _count: int

    def __init__(
            self,
            transitions: List[TransitionWithDiscount],
            batch_size: int,
            n_steps: int = 1,
            gamma: float = 0.99,
            n_frames: int = 1,
            real_ratio: float = 1.0,
            generated_maxlen: int = 100000,
    ):
        self._transitions = transitions
        self._generated_transitions = FIFOQueue(generated_maxlen)
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._gamma = gamma
        self._n_frames = n_frames
        self._real_ratio = real_ratio
        self._real_batch_size = batch_size
        self._count = 0

    def __iter__(self) -> Iterator[TransitionMiniBatchWithDiscount]:
        self.reset()
        return self

    def __next__(self) -> TransitionMiniBatchWithDiscount:
        if len(self._generated_transitions) > 0:
            real_batch_size = self._real_batch_size
            fake_batch_size = self._batch_size - self._real_batch_size
            transitions_w_discounts = [self.get_next() for _ in range(real_batch_size)]
            transitions_w_discounts += self._sample_generated_transitions(fake_batch_size)
        else:
            transitions_w_discounts = [self.get_next() for _ in range(self._batch_size)]

        transitions = [t.transition for t in transitions_w_discounts]
        discounts = [t.discount for t in transitions_w_discounts]

        batch = TransitionMiniBatch(
            transitions,
            n_frames=self._n_frames,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )

        self._count += 1

        return TransitionMiniBatchWithDiscount(batch, np.array(discounts))

    def _reset(self) -> None:
        pass

    def reset(self) -> None:
        self._count = 0
        if len(self._generated_transitions) > 0:
            self._real_batch_size = int(self._real_ratio * self._batch_size)
        self._reset()

    @abstractmethod
    def _next(self) -> TransitionWithDiscount:
        pass

    @abstractmethod
    def _has_finished(self) -> bool:
        pass

    def add_generated_transitions(self, transitions: List[TransitionWithDiscount]) -> None:
        self._generated_transitions.extend(transitions)

    def get_next(self) -> TransitionWithDiscount:
        if self._has_finished():
            raise StopIteration
        return self._next()

    def _sample_generated_transitions(
            self, batch_size: int
    ) -> List[TransitionWithDiscount]:
        transitions: List[TransitionWithDiscount] = []
        n_generated_transitions = len(self._generated_transitions)
        for _ in range(batch_size):
            index = cast(int, np.random.randint(n_generated_transitions))
            transitions.append(self._generated_transitions[index])
        return transitions

    @abstractmethod
    def __len__(self) -> int:
        pass

    def size(self) -> int:
        return len(self._transitions) + len(self._generated_transitions)

    @property
    def transitions(self) -> List[TransitionWithDiscount]:
        return self._transitions

    @property
    def generated_transitions(self) -> FIFOQueue[TransitionWithDiscount]:
        return self._generated_transitions


class DiscountTransitionRandomIterator(DiscountTransitionIterator):
    _n_steps_per_epoch: int

    def __init__(
            self,
            transitions: List[TransitionWithDiscount],
            n_steps_per_epoch: int,
            batch_size: int,
            n_steps: int = 1,
            gamma: float = 0.99,
            n_frames: int = 1,
            real_ratio: float = 1.0,
            generated_maxlen: int = 100000,
    ):
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._n_steps_per_epoch = n_steps_per_epoch

    def _reset(self) -> None:
        pass

    def _next(self) -> TransitionWithDiscount:
        index = cast(int, np.random.randint(len(self._transitions)))
        transition = self._transitions[index]
        return transition

    def _has_finished(self) -> bool:
        return self._count >= self._n_steps_per_epoch

    def __len__(self) -> int:
        return self._n_steps_per_epoch

class DiscountTransitionRoundIterator(DiscountTransitionIterator):

    _shuffle: bool
    _indices: np.ndarray
    _index: int

    def __init__(
        self,
        transitions: List[TransitionWithDiscount],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
        shuffle: bool = True,
    ):
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._shuffle = shuffle
        self._indices = np.arange(len(self._transitions))
        self._index = 0

    def _reset(self) -> None:
        self._indices = np.arange(len(self._transitions))
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._index = 0

    def _next(self) -> TransitionWithDiscount:
        transition = self._transitions[cast(int, self._indices[self._index])]
        self._index += 1
        return transition

    def _has_finished(self) -> bool:
        return self._index >= len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions) // self._real_batch_size


class InitialStepRandomIterator(DiscountTransitionRandomIterator):
    """
    There are fewer initial states. So we should use random iterator instead of round iterator
    """

    def __init__(self, dataset: ProbabilityMDPDataset, batch_size: int,
                 gamma: float = 1.0,
                 n_steps_per_epoch: int = 10000):
        init_trans = []

        for ep in dataset.episodes:
            transitions_with_discounts = []
            discount = 1
            for step_t, tr in enumerate(ep.transitions):
                transitions_with_discounts.append(TransitionWithDiscount(
                    tr, discount))
                discount *= gamma

            init_trans.extend(transitions_with_discounts)

        super().__init__(init_trans, batch_size=batch_size, n_steps_per_epoch=n_steps_per_epoch)

class InitialStepRoundIterator(DiscountTransitionRoundIterator):
    """
    There are fewer initial states. So we should use random iterator instead of round iterator
    """

    def __init__(self, dataset: ProbabilityMDPDataset, batch_size: int,
                 gamma: float = 1.0):
        init_trans = []

        for ep in dataset.episodes:
            transitions_with_discounts = []
            discount = 1
            for step_t, tr in enumerate(ep.transitions):
                transitions_with_discounts.append(TransitionWithDiscount(
                    tr, discount))
                discount *= gamma

            init_trans.extend(transitions_with_discounts)

        super().__init__(init_trans, batch_size=batch_size)


class ConsecutiveTransitionIterator(metaclass=ABCMeta):
    _transitions: List[Tuple[TransitionWithDiscount, TransitionWithDiscount]]
    _generated_transitions: FIFOQueue[Tuple[TransitionWithDiscount, TransitionWithDiscount]]
    _batch_size: int
    _n_steps: int
    _gamma: float
    _n_frames: int
    _real_ratio: float
    _real_batch_size: int
    _count: int

    def __init__(
            self,
            transitions: List[Tuple[TransitionWithDiscount, TransitionWithDiscount]],
            batch_size: int,
            n_steps: int = 1,
            gamma: float = 0.99,
            real_ratio: float = 1.0,
            generated_maxlen: int = 100000,
    ):
        self._transitions = transitions
        self._generated_transitions = FIFOQueue(generated_maxlen)
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._gamma = gamma
        self._real_ratio = real_ratio
        self._real_batch_size = batch_size
        self._count = 0
        self._n_frames = 1

    def __iter__(self) -> Iterator[Tuple[TransitionMiniBatchWithDiscount, TransitionMiniBatchWithDiscount]]:
        self.reset()
        return self

    def __next__(self) -> Tuple[TransitionMiniBatchWithDiscount, TransitionMiniBatchWithDiscount]:
        if len(self._generated_transitions) > 0:
            real_batch_size = self._real_batch_size
            fake_batch_size = self._batch_size - self._real_batch_size
            transitions = [self.get_next() for _ in range(real_batch_size)]
            transitions += self._sample_generated_transitions(fake_batch_size)
        else:
            transitions = [self.get_next() for _ in range(self._batch_size)]

        first_transition_steps = [t[0].transition for t in transitions]
        first_discount_steps = [t[0].discount for t in transitions]
        second_transition_steps = [t[1].transition for t in transitions]
        second_discount_steps = [t[1].discount for t in transitions]

        batch1 = TransitionMiniBatch(
            first_transition_steps,
            n_frames=self._n_frames,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )

        batch2 = TransitionMiniBatch(
            second_transition_steps,
            n_frames=self._n_frames,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )

        self._count += 1

        return TransitionMiniBatchWithDiscount(batch1, np.array(first_discount_steps)), \
               TransitionMiniBatchWithDiscount(batch2, np.array(second_discount_steps))

    def reset(self) -> None:
        self._count = 0
        if len(self._generated_transitions) > 0:
            self._real_batch_size = int(self._real_ratio * self._batch_size)
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        pass

    @abstractmethod
    def _next(self) -> Tuple[TransitionWithDiscount, TransitionWithDiscount]:
        pass

    @abstractmethod
    def _has_finished(self) -> bool:
        pass

    def add_generated_transitions(self,
                                  transitions: List[Tuple[TransitionWithDiscount, TransitionWithDiscount]]) -> None:
        self._generated_transitions.extend(transitions)

    def get_next(self) -> Tuple[TransitionWithDiscount, TransitionWithDiscount]:
        if self._has_finished():
            raise StopIteration
        return self._next()

    def _sample_generated_transitions(
            self, batch_size: int
    ) -> List[Tuple[TransitionWithDiscount, TransitionWithDiscount]]:
        transitions: List[Tuple[TransitionWithDiscount, TransitionWithDiscount]] = []
        n_generated_transitions = len(self._generated_transitions)
        for _ in range(batch_size):
            index = cast(int, np.random.randint(n_generated_transitions))
            transitions.append(self._generated_transitions[index])
        return transitions

    @abstractmethod
    def __len__(self) -> int:
        pass

    def size(self) -> int:
        return len(self._transitions) + len(self._generated_transitions)

    @property
    def transitions(self) -> List[Tuple[TransitionWithDiscount, TransitionWithDiscount]]:
        return self._transitions

    @property
    def generated_transitions(self) -> FIFOQueue[Tuple[TransitionWithDiscount, TransitionWithDiscount]]:
        return self._generated_transitions


class ConsecutiveRoundIterator(ConsecutiveTransitionIterator):
    _shuffle: bool
    _indices: np.ndarray
    _index: int

    def __init__(
            self,
            transitions: List[Tuple[TransitionWithDiscount, TransitionWithDiscount]],
            batch_size: int,
            n_steps: int = 1,
            gamma: float = 0.99,
            real_ratio: float = 1.0,
            generated_maxlen: int = 100000,
            shuffle: bool = True,
    ):
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._shuffle = shuffle
        self._indices = np.arange(len(self._transitions))
        self._index = 0

    def _reset(self) -> None:
        self._indices = np.arange(len(self._transitions))
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._index = 0

    def _next(self) -> Tuple[TransitionWithDiscount, TransitionWithDiscount]:
        transition = self._transitions[cast(int, self._indices[self._index])]
        self._index += 1
        return transition

    def _has_finished(self) -> bool:
        return self._index >= len(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions) // self._real_batch_size


class StepNextStepRoundIterator(ConsecutiveRoundIterator):
    def __init__(self, dataset: ProbabilityMDPDataset,
                 batch_size: int, shuffle: bool = True,
                 gamma: float = 1):
        init_trans = []

        for ep in dataset.episodes:
            transitions_with_discounts = []
            discount = 1
            for step_t, tr in enumerate(ep.transitions):
                transitions_with_discounts.append(TransitionWithDiscount(
                    tr, discount))
                discount *= gamma
            init_trans.extend(self.pair_consecutive_elements(transitions_with_discounts))

        # We checked if it's consecutive, it's correct
        # can write unit test here

        super().__init__(init_trans, batch_size, shuffle)

    def pair_consecutive_elements(self, lst: List[TransitionWithDiscount]) -> List[
        Tuple[TransitionWithDiscount, TransitionWithDiscount]]:
        result = [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]
        return result


if __name__ == '__main__':
    from offline_rl.envs.datasets import get_sepsis

    dataset, sepsis = get_sepsis('pomdp-200')
    init_iterator = InitialStepRandomIterator(dataset, 32)
    iterator = StepNextStepRoundIterator(dataset, 4)
    for step, next_step in iterator:
        print(step.transitions.observations)
        print(next_step.transitions.observations)
        break
