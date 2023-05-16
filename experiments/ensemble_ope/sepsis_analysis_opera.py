"""
We produce the stats files necessary for Sepsis analysis

We just load in the bootstrap result from many datasets and run OPERA

The main goal is to sweep over all n
load in true MSE, bootstrap MSE
"""
from collections import defaultdict

from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
    get_sepsis_subsample_copies, get_sepsis_population_full, get_sepsis_gt, get_sepsis_copies
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF, \
    load_sepsis_ensemble_mdp_policies, MDP_POLICY_TURE_PERF

from offline_rl.envs.dataset import convert_dataset_for_is_ope
from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
    cwpdis_scorer, pdis_scorer

from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np

import pickle


def solve_for_alpha(ope_scores, error_matrix_A):
    n = error_matrix_A.shape[0]

    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(x, error_matrix_A))
    constraints = [cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    print('cvxpy loss: ', prob.solve())

    alpha = x.value.flatten()

    score = (ope_scores * alpha).sum()

    return alpha, score


def load_true_scores(n, estimator_name, env_name):
    assert env_name in ['mdp', 'pomdp']
    assert estimator_name in ['IS', 'WIS', 'CLIS', 'CLWIS']
    file_dir = "./sepsis_analysis_results/"
    if env_name == 'mdp':
        file_name = 'env_mdp_true_MSE_{}_n_{}.npz'
    else:
        file_name = 'env_pomdp_true_MSE_{}_n_{}.npz'
    # this is actually the true score
    # (dataset_sample_times,)
    return np.load(file_dir + file_name.format(estimator_name, n))['true_MSE']


def load_boostrap_scores(n, estimator_name):
    assert estimator_name in ['IS', 'WIS', 'CLIS', 'CLWIS']
    file_dir = "./sepsis_analysis_results/"
    file_name = 'env_pomdp_bootstrap_MSE_{}_n_{}.npz'
    # (dataset_sample_times, bootstrap_times)
    results = np.load(file_dir + file_name.format(estimator_name, n))
    return results['bootstrap_stats'], results['ope_scores']


def compute_mse_matrix(ope_scores, ope_bootstrapped_scores, n):
    # ope_scores: (n_estimators)
    # ope_bootstrapped_scores: (n_estimators, n_copies) (this can also be MSE)

    # this is what we fixed for Sepsis (should be updated if needed)
    k = int(n ** 0.6)
    scale_ratio = k / n

    n_estimators = ope_scores.shape[0]
    num_bootstrap = ope_bootstrapped_scores.shape[1]
    pre_A = np.zeros((n_estimators, num_bootstrap))
    est_variance = np.zeros(n_estimators)
    est_bias = np.zeros(n_estimators)

    # not adding the scale_ratio (we don't have n, k yet, and it doesn't change optimization, only helps MSE)

    for j in range(n_estimators):
        # est_bias = ope_bootstrapped_scores[j, :].mean() - ope_scores[j]
        est_bias[j] = (ope_bootstrapped_scores[j, :] - ope_scores[j]).mean()
        est_variance[j] = np.mean((ope_bootstrapped_scores[j, :] - ope_scores[j]) ** 2)
        pre_A[j, :] = ope_bootstrapped_scores[j, :] - ope_scores[j]

    error_matrix_A = scale_ratio * (1 / num_bootstrap) * np.matmul(pre_A, pre_A.T)
    ope_mse = np.diagonal(error_matrix_A)

    return error_matrix_A, ope_mse, est_bias, est_variance


def compute_opera_scores(true_mdp_sample_times=50, bootstrap_mdp_sample_times=20, n_copies=100):
    n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                1100, 1200, 1300, 1400, 1500]  # 1600, 2400, 3200

    # we should compute stats for ALL estimators (MSE)
    # save the following things:
    # 1. MSE per estimator for all ns (bootstrap should be lower than true MSE)
    # 2. alphas outputted by ours for all estimators
    # 3. SwitchOPE, AverageOPE, etc.

    estimator_names = ['IS', 'WIS']  # 'CLIS', 'CLWIS'

    true_mse = {}
    true_bias, true_variance = defaultdict(list), defaultdict(list)

    # let's first get stats per estimator
    for estimator_name in tqdm(estimator_names):
        true_mse[estimator_name] = []
        for n in n_values:
            true_scores = load_true_scores(n, estimator_name, 'pomdp')
            mse = ((true_scores - opt_true_perf) ** 2).mean()
            true_mse[estimator_name].append(mse)
            bias = (true_scores - opt_true_perf).mean()
            variance = ((true_scores - np.mean(true_scores)) ** 2).mean()
            true_bias[estimator_name].append(bias)
            true_variance[estimator_name].append(variance)

    # then let's do bootstrap_MSE
    bootstrap_mse = {}
    bootstrap_bias, bootstrap_variance = defaultdict(list), defaultdict(list)
    for estimator_name in tqdm(estimator_names):
        bootstrap_mse[estimator_name] = []
        for n in n_values:
            bootstrap_scores, ope_scores = load_boostrap_scores(n, estimator_name)
            # we additionally have to iterate through the sample_time here
            average_mse = []
            average_bias, average_variance = [], []
            for i in range(bootstrap_scores.shape[0]):
                # this is all bootstrap copy on one dataset
                mse = ((bootstrap_scores[i, :] - opt_true_perf) ** 2).mean()
                average_mse.append(mse)
                bias = (bootstrap_scores[i, :] - opt_true_perf).mean()
                variance = ((bootstrap_scores[i, :] - np.mean(bootstrap_scores[i, :])) ** 2).mean()
                average_bias.append(bias)
                average_variance.append(variance)

            bootstrap_mse[estimator_name].append(np.mean(average_mse))
            bootstrap_bias[estimator_name].append(np.mean(average_bias))
            bootstrap_variance[estimator_name].append(np.mean(average_variance))

    # then let's do OPERA, use true MSE, use bootstrap MSE
    opera_running_stats = {}

    # first true MSE
    opera_running_stats['true_mse'] = {}
    opera_true_mse, opera_true_bias = [], []
    opera_scores = []
    opera_alphas = []
    for n in tqdm(n_values):
        # we use the average of true_scores as the OPE score
        ope_scores = np.zeros(len(estimator_names))
        ope_bootstrapped_scores = np.zeros((len(estimator_names), true_mdp_sample_times))
        for j, estimator_name in enumerate(estimator_names):
            true_scores = load_true_scores(n, estimator_name, 'pomdp')
            ope_scores[j] = true_scores.mean()
            ope_bootstrapped_scores[j, :] = true_scores

        error_matrix_A, ope_mse, est_bias, est_variance = compute_mse_matrix(ope_scores, ope_bootstrapped_scores, n)
        alpha, score = solve_for_alpha(ope_scores, error_matrix_A)

        # this is SE, not MSE
        opera_true_mse.append((score - opt_true_perf) ** 2)
        opera_true_bias.append(score - opt_true_perf)

        opera_scores.append(score)
        opera_alphas.append(alpha)

    opera_running_stats['true_mse']['scores'] = opera_scores
    opera_running_stats['true_mse']['alphas'] = opera_alphas
    true_mse['opera'] = opera_true_mse
    true_bias['opera'] = opera_true_bias

    # then bootstrap MSE
    opera_running_stats['bootstrap_mse'] = {}
    opera_true_mse, opera_true_bias, opera_true_variance = [], [], []
    for n in tqdm(n_values):
        scores = np.zeros(bootstrap_mdp_sample_times)
        # essentially we ran our algorithm on 20 datasets
        for i in range(bootstrap_mdp_sample_times):
            ope_scores = np.zeros(len(estimator_names))
            ope_bootstrapped_scores = np.zeros((len(estimator_names), n_copies))
            for j, estimator_name in enumerate(estimator_names):
                bootstrap_scores, scores = load_boostrap_scores(n, estimator_name)
                ope_scores[j] = scores[i]
                ope_bootstrapped_scores[j, :] = bootstrap_scores[i, :] # i-th dataset for j-th estimator

            error_matrix_A, ope_mse, est_bias, est_variance = compute_mse_matrix(ope_scores, ope_bootstrapped_scores, n)
            alpha, score = solve_for_alpha(ope_scores, error_matrix_A)
            scores[i] = score

        mse = ((scores - opt_true_perf) ** 2).mean()
        bias = (scores - opt_true_perf).mean()
        variance = ((scores - np.mean(scores)) ** 2).mean()

        opera_true_mse.append(mse)
        opera_true_bias.append(bias)
        opera_true_variance.append(variance)

    bootstrap_mse['opera'] = opera_true_mse
    bootstrap_bias['opera'] = opera_true_bias
    bootstrap_variance['opera'] = opera_true_variance

    pickle.dump(true_mse, open('sepsis_analysis_stats/true_mse.pkl', 'wb'))
    pickle.dump(true_bias, open('sepsis_analysis_stats/true_bias.pkl', 'wb'))
    pickle.dump(true_variance, open('sepsis_analysis_stats/true_variance.pkl', 'wb'))
    pickle.dump(bootstrap_mse, open('sepsis_analysis_stats/bootstrap_mse.pkl', 'wb'))
    pickle.dump(bootstrap_bias, open('sepsis_analysis_stats/bootstrap_bias.pkl', 'wb'))
    pickle.dump(bootstrap_variance, open('sepsis_analysis_stats/bootstrap_variance.pkl', 'wb'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="sepsis_analysis_results")
    parser.add_argument('--env', type=str, default="pomdp")

    # use True MSE
    parser.add_argument('--use_true_mse', action='store_true')
    parser.add_argument('--n', type=int, default=100)

    # use Bootstrap MSE
    parser.add_argument('--use_bootstrap_mse', action='store_true')
    parser.add_argument('--n_copies', type=int, default=100)
    parser.add_argument('--sample_times', type=int, default=20)

    args = parser.parse_args()
    # Plan:
    # we will use 4 estimators: IS, WIS, Clipped IS, Clipped WIS
    # to basically show trade-offs

    print(args)

    # ==== Load the policy ====
    dataset, sepsis, data = get_sepsis_gt('{}-200'.format(args.env))
    if args.env == 'pomdp':
        policies = load_sepsis_ensemble_policies(sepsis)
    else:
        policies = load_sepsis_ensemble_mdp_policies(sepsis)

    opt_policy = policies[0]
    opt_true_perf = POLICY_TRUE_MEAN_PERF[0] if args.env == 'pomdp' else MDP_POLICY_TURE_PERF[0]

    # ===== End =======
    compute_opera_scores()