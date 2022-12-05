from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, \
                   NamedTuple
import torch
import numpy as np


class StepType(object):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = np.asarray(0, dtype=np.int32)
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = np.asarray(1, dtype=np.int32)
  # Denotes the last `TimeStep` in a sequence.
  LAST = np.asarray(2, dtype=np.int32)

  def __new__(cls, value):
    """Add ability to create StepType constants from a value."""
    if value == cls.FIRST:
      return cls.FIRST
    if value == cls.MID:
      return cls.MID
    if value == cls.LAST:
      return cls.LAST

    raise ValueError('No known conversion for `%r` into a StepType' % value)

# TODO: need to check if the function returns bool or not
class TimeStep(
    NamedTuple('TimeStep', [('step_type', StepType),
                            ('reward', torch.Tensor),
                            ('discount', torch.Tensor),
                            ('observation', np.ndarray)])):
  """Returned with every call to `step` and `reset` on an environment.

  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
  NumPy array or a dict or list of arrays), and an associated `reward` and
  `discount`.

  The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
  `TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
  will equal `StepType.MID.

  Attributes:
    step_type: a `Tensor` or array of `StepType` enum values.
    reward: a `Tensor` or array of reward values.
    discount: A discount value in the range `[0, 1]`.
    observation: A NumPy array, or a nested dict, list or tuple of arrays.
  """
  __slots__ = ()

  def is_first(self) -> bool:
    # if torch.is_tensor(self.step_type):
    #   return torch.equal(self.step_type, StepType.FIRST)
    # return np.equal(self.step_type, StepType.FIRST)
    return self.step_type == StepType.FIRST

  def is_mid(self) -> bool:
    # if tf.is_tensor(self.step_type):
    #   return tf.equal(self.step_type, StepType.MID)
    # return np.equal(self.step_type, StepType.MID)

    return self.step_type == StepType.FIRST

  def is_last(self) -> bool:
    # if tf.is_tensor(self.step_type):
    #   return tf.equal(self.step_type, StepType.LAST)
    # return np.equal(self.step_type, StepType.LAST)
    return self.step_type == StepType.FIRST

  def __hash__(self):
    # TODO(b/130243327): Explore performance impact and consider converting
    # dicts in the observation into ordered dicts in __new__ call.
    return hash(tuple(self))

