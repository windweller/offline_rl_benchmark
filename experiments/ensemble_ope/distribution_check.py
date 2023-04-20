from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
                                    get_sepsis_subsample_copies, get_sepsis_weighted, get_sepsis_population_full
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF

from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment
from offline_rl.envs.dataset import convert_dataset_for_is_ope
from sklearn.model_selection import train_test_split

from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer

from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
    cwpdis_scorer, pdis_scorer
from offline_rl.opes.regression_importance_sampling import RegressionIS

from d3rlpy.algos import DQN, DiscreteCQL, DiscreteBC

from tqdm import tqdm
import pandas as pd

import cvxpy as cp
import numpy as np

from offline_rl.envs.dataset import ProbabilityMDPDataset
from offline_rl.algs.policy_evaluation_wrappers import DiscreteProbabilisticPolicyProtocol

def calculate_scores(evaluation_policy, datasets,
                      ope_scorers=[], ris=False):
    # This doesn't work because we need to bootstrap actual patient trajectories
    # not individual states

    n = len(ope_scorers)
    mat_n = n if ris is False else n * 2
    ope_scores = np.zeros((len(datasets), mat_n))

    sub_optimal_policy = evaluation_policy

    for j, _ in enumerate(ope_scorers):

        for i, dataset in tqdm(enumerate(datasets)):
            dataset_with_prob = convert_dataset_for_is_ope(dataset)
            clipped_is_score = ope_scorers[j](sub_optimal_policy, dataset_with_prob.episodes)
            ope_scores[i, j] = clipped_is_score

    if ris:
        for j, _ in enumerate(ope_scorers):

            for i, dataset in tqdm(enumerate(datasets)):
                # train behavior policy
                bc = DiscreteBC(use_gpu=False)
                bc.build_with_dataset(dataset)
                ris = RegressionIS(bc)
                ris.fit(dataset, n_epochs=5)
                prob_dataset = ris.estimate_bh_probabilities(dataset)

                dataset_with_prob = convert_dataset_for_is_ope(prob_dataset)

                clipped_is_score = ope_scorers[j](sub_optimal_policy, dataset_with_prob.episodes)
                ope_scores[i, j+n] = clipped_is_score

    return ope_scores

def record_distribution(dataset_name, ris=False):
    # Load in 100 copies of the dataset, and evaluate OPE on them
    # get the distribution of the OPE scores
    # for each policy

    datasets, sepsis = get_sepsis_population_full(dataset_name)
    policies = load_sepsis_ensemble_policies(sepsis)

    # delete the behavior policy we used to sample data
    del policies[1]

    print(len(datasets))

    for i in tqdm(range(len(policies))):

        ope_scores = calculate_scores(policies[i], datasets, ope_scorers=[importance_sampling_scorer, wis_scorer], ris=ris)
        np.savez(f"results/true_sample_policy_{i}_{dataset_name}_ope_scores.npz", ope_scores=ope_scores)

if __name__ == '__main__':
    # record_distribution("pomdp-200")
    record_distribution("pomdp-1000")