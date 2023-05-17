"""
We produce the stats files necessary for Sepsis analysis
"""

from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
    get_sepsis_subsample_copies, get_sepsis_population_full, get_sepsis_gt, get_sepsis_copies
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF, \
    load_sepsis_ensemble_mdp_policies, MDP_POLICY_TURE_MEAN_PERF

from offline_rl.envs.dataset import convert_dataset_for_is_ope
from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
    cwpdis_scorer, pdis_scorer, importance_sampling_scorer_with_weights

from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np


# NOTE: this new experiment is w.r.t. noise 005 policy and collection
# We can do a mixture of 005 and 000 trajectory too if this doesn't work

# Once we save these, actually, we can get

def save_true_MSE_for_scorer(env, n, scorer, scorer_name, save_dir):
    assert env in {'pomdp', 'mdp'}
    sample_times = 50

    true_MSE = np.zeros(sample_times)
    for j in tqdm(range(sample_times)):
        dataset, sepsis, _ = get_sepsis_gt(env, n)
        dataset_with_prob = convert_dataset_for_is_ope(dataset)
        score = scorer(opt_policy, dataset_with_prob.episodes)
        true_MSE[j] = score

    np.savez(f'{save_dir}/env_{env}_true_MSE_{scorer_name}_n_{n}.npz', true_MSE=true_MSE, sample_times=sample_times)


def save_bootstrap_MSE_for_scorer(env, n, n_copies, sample_times, scorer, scorer_name, save_dir):
    assert env in {'pomdp', 'mdp'}
    bootstrap_stats = np.zeros((sample_times, n_copies))
    ope_scores = np.zeros((sample_times))

    for d_i in tqdm(range(sample_times)):
        dataset, sepsis, pd_data = get_sepsis_gt(env, n)
        dataset_with_prob = convert_dataset_for_is_ope(dataset)
        score = scorer(opt_policy, dataset_with_prob.episodes)
        ope_scores[d_i] = score

        k = int(n ** 0.6)
        k_prop = k / n
        datasets, sepsis, traj_dist, scale_ratio = get_sepsis_copies(pd_data, sepsis, n_copies, k_prop=k_prop)
        for j, dataset in enumerate(datasets):
            dataset_with_prob = convert_dataset_for_is_ope(dataset)
            score = scorer(opt_policy, dataset_with_prob.episodes)
            bootstrap_stats[d_i, j] = score

    np.savez(f'{save_dir}/env_{env}_bootstrap_MSE_{scorer_name}_n_{n}.npz', bootstrap_stats=bootstrap_stats,
             n_copies=n_copies, sample_times=sample_times, scale_ratio=scale_ratio, ope_scores=ope_scores)


# this is a partial function
def no_clip_scorer(scorer_fn):
    return lambda policy, episodes: scorer_fn(policy, episodes, no_clip=True)


def clip_scorer(scorer_fn, clip_ratio):
    upper = 1 / clip_ratio
    lower = clip_ratio
    return lambda policy, episodes: scorer_fn(policy, episodes, clip_lower=lower, clip_upper=upper, no_clip=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="sepsis_analysis_results")
    parser.add_argument('--env', type=str, default="pomdp")
    parser.add_argument('--scorer', type=str, default='IS', choices=['IS', 'WIS', 'CLIS', 'CLWIS'])
    parser.add_argument('--clip', type=float, default=0.02)

    # for True MSE
    parser.add_argument('--compute_true_mse', action='store_true')
    # MAGIC doesn't use bootstrap, so we only need to save it for True MSE case
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
    opt_true_perf = POLICY_TRUE_MEAN_PERF[0] if args.env == 'pomdp' else MDP_POLICY_TURE_MEAN_PERF[0]

    # ===== End =======

    if args.scorer == 'IS':
        scorer = no_clip_scorer(importance_sampling_scorer)
    elif args.scorer == 'WIS':
        scorer = no_clip_scorer(wis_scorer)
    elif args.scorer == 'CLIS':
        scorer = clip_scorer(importance_sampling_scorer, args.clip)
    elif args.scorer == 'CLWIS':
        scorer = clip_scorer(wis_scorer, args.clip)
    else:
        raise NotImplementedError

    if args.compute_true_mse:
        save_true_MSE_for_scorer(args.env, args.n, scorer, args.scorer, args.save_dir)

    if args.compute_bootstrap_mse:
        save_bootstrap_MSE_for_scorer(args.env, args.n, args.n_copies, args.sample_times, scorer, args.scorer,
                                      args.save_dir)
