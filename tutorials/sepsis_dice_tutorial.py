from offline_rl.opes.dice_rl.data.dataset import InitialStepRandomIterator, StepNextStepRoundIterator

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC
from offline_rl.envs.datasets import get_sepsis

cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('pomdp-200')
cql.build_with_dataset(dataset)

init_iterator = InitialStepRandomIterator(dataset, 32)
iterator = StepNextStepRoundIterator(dataset, 32)

