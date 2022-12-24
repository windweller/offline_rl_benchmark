"""
RIS(0) estimator -- we use BC to estimate probability,
and then use it with IS OPE estimators
"""
from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC
from offline_rl.envs.datasets import get_sepsis
from offline_rl.algs.policy_evaluation_wrappers import DiscreteCQLEvaluationWrapper
from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment
from offline_rl.envs.dataset import convert_dataset_for_is_ope, convert_is_ope_dataset_for_training
from sklearn.model_selection import train_test_split

from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
            cwpdis_scorer, pdis_scorer
from offline_rl.opes.regression_importance_sampling import RegressionIS

cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('pomdp-200')
# cql.build_with_env(sepsis)
cql.build_with_dataset(dataset)

evaluation_policy = DiscreteCQLEvaluationWrapper(cql)
dataset_with_prob = convert_dataset_for_is_ope(dataset)

clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
weighted_is_score = wis_scorer(evaluation_policy, dataset_with_prob.episodes)
cwpdis_score = cwpdis_scorer(evaluation_policy, dataset_with_prob.episodes)
pdis_score = pdis_scorer(evaluation_policy, dataset_with_prob.episodes)

print("OIS:")
print(clipped_is_score)
print(weighted_is_score)
print(cwpdis_score)
print(pdis_score)

# now we do RIS
bc = DiscreteBC(use_gpu=False)
bc.build_with_dataset(dataset)
ris = RegressionIS(bc)
ris.fit(dataset, n_epochs=5)
prob_dataset = ris.estimate_bh_probabilities(dataset)

dataset_with_prob = convert_dataset_for_is_ope(prob_dataset)

clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
weighted_is_score = wis_scorer(evaluation_policy, dataset_with_prob.episodes)
cwpdis_score = cwpdis_scorer(evaluation_policy, dataset_with_prob.episodes)
pdis_score = pdis_scorer(evaluation_policy, dataset_with_prob.episodes)

print("RIS:")
print(clipped_is_score)
print(weighted_is_score)
print(cwpdis_score)
print(pdis_score)