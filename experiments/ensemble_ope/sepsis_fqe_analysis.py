"""
We produce the stats files necessary for Sepsis analysis
"""

from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
    get_sepsis_subsample_copies, get_sepsis_population_full, get_sepsis_gt, get_sepsis_copies
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF, \
    load_sepsis_ensemble_mdp_policies, MDP_POLICY_TURE_PERF

from offline_rl.opes.tabular_fqe import TabularFQE
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np

# ==== Load the policy ====
dataset, sepsis, data = get_sepsis_gt('pomdp-200')
policies = load_sepsis_ensemble_policies(sepsis)

opt_policy = policies[0]
opt_true_perf = POLICY_TRUE_MEAN_PERF[0]


# ===== End =======

# NOTE: this new experiment is w.r.t. noise 005 policy and collection
# We can do a mixture of 005 and 000 trajectory too if this doesn't work

# Once we save these, actually, we can get

def cross_fit_fqe_scorer(sepsis, dataset):
    # cross-fitting (in causal inference)
    scores = []
    train_episodes, test_episodes = train_test_split(dataset.episodes, test_size=0.5)

    fqe = TabularFQE(algo=opt_policy)
    fqe.build_with_env(sepsis)

    fqe.fit(train_episodes, n_epochs=10)
    score = initial_state_value_estimation_scorer(fqe, test_episodes)
    scores.append(score)

    fqe = TabularFQE(algo=opt_policy)
    fqe.build_with_env(sepsis)

    fqe.fit(test_episodes, n_epochs=10)
    score = initial_state_value_estimation_scorer(fqe, train_episodes)
    scores.append(score)

    score = sum(scores) / 2

    return score


def save_true_MSE_for_scorer(env, n, scorer_name, save_dir):
    assert env in {'pomdp', 'mdp'}
    sample_times = 50

    true_MSE = np.zeros(sample_times)
    for j in tqdm(range(sample_times)):
        dataset, sepsis, _ = get_sepsis_gt(env, n)
        true_MSE[j] = cross_fit_fqe_scorer(sepsis, dataset)

    np.savez(f'{save_dir}/env_{env}_true_MSE_{scorer_name}_n_{n}.npz', true_MSE=true_MSE, sample_times=sample_times)


def save_bootstrap_MSE_for_scorer(env, n, n_copies, sample_times, scorer_name, save_dir):
    assert env in {'pomdp', 'mdp'}
    bootstrap_stats = np.zeros((sample_times, n_copies))

    for d_i in tqdm(range(sample_times)):
        dataset, sepsis, pd_data = get_sepsis_gt(env, n)
        k = int(n ** 0.6)
        k_prop = k / n
        datasets, sepsis, traj_dist, scale_ratio = get_sepsis_copies(pd_data, sepsis, n_copies, k_prop=k_prop)
        for j, dataset in enumerate(datasets):
            score = cross_fit_fqe_scorer(sepsis, dataset)
            bootstrap_stats[d_i, j] = score

    np.savez(f'{save_dir}/env_{env}_bootstrap_MSE_{scorer_name}_n_{n}.npz', bootstrap_stats=bootstrap_stats,
             n_copies=n_copies,
             sample_times=sample_times)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="sepsis_analysis_results")
    parser.add_argument('--env', type=str, default="pomdp")
    parser.add_argument('--scorer', type=str, default='IS', choices=['FQE'])
    parser.add_argument('--clip', type=float, default=0.02)

    # for True MSE
    parser.add_argument('--compute_true_mse', action='store_true')
    parser.add_argument('--n', type=int, default=100)

    # for Bootstrap MSE
    parser.add_argument('--compute_bootstrap_mse', action='store_true')
    # parser.add_argument('--n', type=int, default=100)
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

    if args.compute_true_mse:
        save_true_MSE_for_scorer(args.env, args.n, args.scorer, args.save_dir)

    if args.compute_bootstrap_mse:
        save_bootstrap_MSE_for_scorer(args.env, args.n, args.n_copies, args.sample_times, args.scorer, args.save_dir)
