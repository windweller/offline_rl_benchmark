"""
We produce the stats files necessary for Sepsis analysis
"""

from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
    get_sepsis_subsample_copies, get_sepsis_population_full, get_sepsis_gt, get_sepsis_copies, \
    get_sepsis_train_test_split
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF, \
    load_sepsis_ensemble_mdp_policies, MDP_POLICY_TURE_MEAN_PERF

from offline_rl.envs.dataset import convert_dataset_for_is_ope
from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
    cwpdis_scorer, pdis_scorer, importance_sampling_scorer_with_weights

from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np

from offline_rl.opes.tabular_fqe import TabularFQE

from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import _make_batches


# NOTE: this new experiment is w.r.t. noise 005 policy and collection
# We can do a mixture of 005 and 000 trajectory too if this doesn't work

# Once we save these, actually, we can get

def initial_state_value_estimation_scorer(
        algo, episodes, return_per_traj=False
):
    total_values = []
    WINDOW_SIZE = 1024
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate action-value in initial states
            actions = algo.predict([batch.observations[0]])
            values = algo.predict_value([batch.observations[0]], actions)
            total_values.append(values[0])

    if return_per_traj:
        return float(np.mean(total_values)), total_values
    else:
        return float(np.mean(total_values))


def cross_fit_fqe_scorer(sepsis, dataset):
    # cross-fitting (in causal inference)
    scores = []
    all_traj_scores = []
    train_episodes, test_episodes = train_test_split(dataset.episodes, test_size=0.5)

    fqe = TabularFQE(algo=opt_policy)
    fqe.build_with_env(sepsis)

    fqe.fit(train_episodes, n_epochs=10)
    score, traj_scores = initial_state_value_estimation_scorer(fqe, test_episodes, return_per_traj=True)
    scores.append(score)
    all_traj_scores.append(traj_scores)

    fqe = TabularFQE(algo=opt_policy)
    fqe.build_with_env(sepsis)

    fqe.fit(test_episodes, n_epochs=10)
    score, traj_scores = initial_state_value_estimation_scorer(fqe, train_episodes, return_per_traj=True)
    scores.append(score)
    all_traj_scores.insert(0, traj_scores)

    all_traj_scores = np.concatenate(all_traj_scores, axis=0)

    score = sum(scores) / 2

    return score, all_traj_scores


def percentile_boostrap(dataset_pd, env, scorer, alpha=0.1):
    wis_scores = []
    datasets, sepsis, traj_dist, scale_ratio = get_sepsis_copies(dataset_pd, env, 100, k_prop=1.0)
    for j, dataset in enumerate(datasets):
        dataset_with_prob = convert_dataset_for_is_ope(dataset)
        score = scorer(opt_policy, dataset_with_prob.episodes)
        wis_scores.append(score)

    lower_percentile = (1 - alpha) / 2
    upper_percentile = 1 - lower_percentile

    return np.percentile(wis_scores, lower_percentile), np.percentile(wis_scores, upper_percentile)


def magic_bias_compute(temp, lb, ub):
    assert ub >= lb
    if temp > ub:
        bias = np.abs(ub - temp)
    elif temp < lb:
        bias = np.abs(lb - temp)
    else:
        bias = 0
    return bias


def k_n_bootstrap(dataset_pd, env, n, num_bootstrap, scorer):
    n_copies = num_bootstrap
    k = int(n ** 0.6)
    k_prop = k / n
    datasets, sepsis, traj_dist, scale_ratio = get_sepsis_copies(dataset_pd, env, n_copies, k_prop=k_prop)

    is_bootstrap_stats, fqe_bootstrap_stats = [], []
    for j, dataset in enumerate(datasets):
        dataset_with_prob = convert_dataset_for_is_ope(dataset)
        score, _ = scorer(opt_policy, dataset_with_prob.episodes)
        is_bootstrap_stats.append(score)

        fqe_score, _ = cross_fit_fqe_scorer(sepsis, dataset)
        fqe_bootstrap_stats.append(fqe_score)

    return np.array(is_bootstrap_stats), np.array(fqe_bootstrap_stats), scale_ratio


def save_true_MSE_for_scorer(env, n, save_dir):
    assert env in {'pomdp', 'mdp'}
    sample_times = 10

    is_scorer = no_clip_scorer(importance_sampling_scorer_with_weights)
    wis_noclip_scorer = no_clip_scorer(wis_scorer)

    true_IS_scores = np.zeros(sample_times)
    true_FQE_scores = np.zeros(sample_times)

    # true_IS_all_trajs = []  # sample_times length
    # true_FQE_all_trajs = []  # sample_times length

    n_estimators = 2
    magic_biases = np.zeros((sample_times, n_estimators))
    magic_mse = np.zeros((sample_times, n_estimators))

    bootstrap_mse = np.zeros((sample_times, n_estimators))

    # Treat this as a trial
    for j in tqdm(range(sample_times)):
        dataset, sepsis, dataset_pd = get_sepsis_gt(env, n)

        # ======== MAGIC =========
        dataset_with_prob = convert_dataset_for_is_ope(dataset)
        is_score, is_weights = is_scorer(opt_policy, dataset_with_prob.episodes)
        fqe_score, fqe_traj = cross_fit_fqe_scorer(sepsis, dataset)

        # get ready to compute magic here,
        # we compute bias and variance for each estimator
        # and compute A here, we then store diagonal of A as our MSE estimation
        delta1 = is_weights - np.mean(is_weights)
        delta2 = fqe_traj - np.mean(fqe_traj)

        delta = np.vstack([delta1, delta2])  # k x n (n = number of traj)
        Omega = np.matmul(delta, delta.T)  # k x k

        lb, ub = percentile_boostrap(dataset_pd, sepsis, wis_noclip_scorer, alpha=0.5)

        temp1 = np.mean(is_weights)
        bias1 = magic_bias_compute(temp1, lb, ub)

        temp2 = np.mean(fqe_traj)
        bias2 = magic_bias_compute(temp2, lb, ub)

        bias = np.array([bias1, bias2])  # k x 1
        b = np.outer(bias, bias)  # k x k

        A = Omega + b  # k x k
        magic_mse[j] = np.diag(A)
        magic_biases[j] = bias

        print("Magic done")

        # ======= Bootstrap MSE ========
        num_bootstrap = 100
        ope_bootstrapped_scores = np.zeros((n_estimators, num_bootstrap))
        is_bootstrap_stats, fqe_bootstrap_stats, scale_ratio = k_n_bootstrap(dataset_pd, sepsis, n, num_bootstrap,
                                                                             is_scorer)
        ope_bootstrapped_scores[0, :] = is_bootstrap_stats
        ope_bootstrapped_scores[1, :] = fqe_bootstrap_stats

        pre_A = np.zeros((n_estimators, num_bootstrap))

        pre_A[0, :] = ope_bootstrapped_scores[0, :] - is_score
        pre_A[1, :] = ope_bootstrapped_scores[1, :] - fqe_score

        error_matrix_A = scale_ratio * (1 / num_bootstrap) * np.matmul(pre_A, pre_A.T)
        bootstrap_mse[j] = np.diag(error_matrix_A)

        print("Bootstrap done")

        true_IS_scores[j] = is_score
        true_FQE_scores[j] = fqe_score

    # ====== true MSE for each estimator =======

    true_IS_mse = np.mean((true_IS_scores - opt_true_perf) ** 2)
    true_FQE_mse = np.mean((true_FQE_scores - opt_true_perf) ** 2)

    np.savez(f'{save_dir}/env_{env}_MSE_MAGIC_n_{n}.npz',
             true_IS_scores=true_IS_scores,
             true_FQE_scores=true_FQE_scores,
             magic_mse=magic_mse,
             magic_biases=magic_biases,
             bootstrap_mse=bootstrap_mse,
             # true_IS_all_trajs=true_IS_all_trajs,
             # true_FQE_all_trajs=true_FQE_all_trajs,
             true_IS_mse=true_IS_mse,
             true_FQE_mse=true_FQE_mse,
             sample_times=sample_times)


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
    parser.add_argument('--save_dir', type=str, default="sepsis_analysis_magic")
    parser.add_argument('--env', type=str, default="mdp")
    parser.add_argument('--clip', type=float, default=0.02)

    # for True MSE
    # parser.add_argument('--compute_true_mse', action='store_true')
    # MAGIC doesn't use bootstrap, so we only need to save it for True MSE case
    parser.add_argument('--n', type=int, default=200)

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

    # for Magic, we need IS, WIS, and FQE
    # they also better be run on the same dataset

    save_true_MSE_for_scorer(args.env, args.n, args.save_dir)
