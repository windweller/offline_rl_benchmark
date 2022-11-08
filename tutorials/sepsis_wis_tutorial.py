# Test WIS-based OPEs

from d3rlpy.algos import DQN, DiscreteCQL
from offline_rl.envs.datasets import get_sepsis
from offline_rl.algs.discrete_policy_evaluation_wrappers import QLearningEvaluationWrapper
from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment
from offline_rl.envs.dataset import convert_dataset_for_is_ope, convert_is_ope_dataset_for_training
from sklearn.model_selection import train_test_split

from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
            cwpdis_scorer, pdis_scorer

cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('pomdp-200')
# cql.build_with_env(sepsis)
cql.build_with_dataset(dataset)

evaluation_policy = QLearningEvaluationWrapper(cql)
dataset_with_prob = convert_dataset_for_is_ope(dataset)

clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
weighted_is_score = wis_scorer(evaluation_policy, dataset_with_prob.episodes)
cwpdis_score = cwpdis_scorer(evaluation_policy, dataset_with_prob.episodes)
pdis_score = pdis_scorer(evaluation_policy, dataset_with_prob.episodes)

print(clipped_is_score)
print(weighted_is_score)
print(cwpdis_score)
print(pdis_score)

# ======= BC =======

# from d3rlpy.algos import BC, DiscreteBC
# from offline_rl.algs.discrete_policy_evaluation_wrappers import DiscreteBCEvaluationWrapper
#
# bc = DiscreteBC(use_gpu=False)
# dataset, sepsis = get_sepsis('pomdp-200')
# # cql.build_with_env(sepsis)
# bc.build_with_dataset(dataset)
#
# evaluation_policy = DiscreteBCEvaluationWrapper(bc)
# dataset_with_prob = convert_dataset_for_is_ope(dataset)
#
# clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
# weighted_is_score = wis_scorer(evaluation_policy, dataset_with_prob.episodes)
# cwpdis_score = cwpdis_scorer(evaluation_policy, dataset_with_prob.episodes)
# pdis_score = pdis_scorer(evaluation_policy, dataset_with_prob.episodes)
#
# print(clipped_is_score)
# print(weighted_is_score)
# print(cwpdis_score)
# print(pdis_score)