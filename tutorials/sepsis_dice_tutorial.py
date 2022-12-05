from offline_rl.opes.dice_rl.data.dataset import InitialStepRandomIterator, StepNextStepRoundIterator

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC
from offline_rl.algs.policy_evaluation_wrappers import DiscreteQLearningTorchWrapper
from offline_rl.envs.datasets import get_sepsis

from offline_rl.opes.dice_rl.estimators.neural_dice import NeuralDice
from offline_rl.opes.dice_rl.networks.value_network import ValueNetwork, ValueNetworkWithAction

from tqdm import tqdm

cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('pomdp-200')
cql.build_with_dataset(dataset)

target_policy = DiscreteQLearningTorchWrapper(cql)

# ===== Training DICE estimator ======

init_iterator = InitialStepRandomIterator(dataset, 32)
iterator = StepNextStepRoundIterator(dataset, 32)

solve_for_state_action_ratio = True
zeta_pos = True
num_steps = 20

if solve_for_state_action_ratio:
    nu_encoder = ValueNetworkWithAction(cql.observation_shape[0] + 1, [128, 128], output_size=1,
                                        zeta_pos=zeta_pos)
    zeta_encoder = ValueNetworkWithAction(cql.observation_shape[0] + 1, [128, 128], output_size=1,
                                        zeta_pos=zeta_pos)
else:
    nu_encoder = ValueNetwork(cql.observation_shape[0], [128, 128], output_size=1, zeta_pos=zeta_pos)
    zeta_encoder = ValueNetwork(cql.observation_shape[0], [128, 128], output_size=1, zeta_pos=zeta_pos)

estimator = NeuralDice(nu_encoder, zeta_encoder, nu_learning_rate=1e-3, zeta_learning_rate=1e-3,
                       gamma=0.99, has_log_probability=True, solve_for_state_action_ratio=solve_for_state_action_ratio,
                       primal_regularizer=0, dual_regularizer=1, norm_regularizer=1, categorical_action=True)

running_losses = []
running_estimates = []

for step in tqdm(range(num_steps)):
    for env_step, next_env_step in iterator:
        initial_steps_batch = next(init_iterator)
        losses = estimator.train_step(initial_steps_batch, env_step, next_env_step, target_policy)
        running_losses.append(losses)

    print(running_losses[-1])

    if step % 500 == 0 or step == num_steps - 1:
      estimate = estimator.estimate_average_reward(dataset, target_policy)
      running_estimates.append(estimate)
      running_losses = []

print(estimate)