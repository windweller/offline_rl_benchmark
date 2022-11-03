# Test WIS-based OPEs

from d3rlpy.algos import DQN, DiscreteCQL
from offline_rl.envs.datasets import get_sepsis
from offline_rl.algs.discrete_policy_evaluation_wrappers import QLearningEvaluationWrapper
from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment, convert_sepsis_dataset_for_is_ope
from sklearn.model_selection import train_test_split

from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, weighted_importance_sampling_scorer, \
            consistently_weighted_per_decision_importance_sampling_scorer

cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('pomdp-200')
# cql.build_with_env(sepsis)
cql.build_with_dataset(dataset)

evaluation_policy = QLearningEvaluationWrapper(cql)
dataset_with_prob = convert_sepsis_dataset_for_is_ope(dataset)

clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
weighted_is_score = weighted_importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
cwpdis_score = consistently_weighted_per_decision_importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)

print(clipped_is_score)
print(weighted_is_score)
print(cwpdis_score)