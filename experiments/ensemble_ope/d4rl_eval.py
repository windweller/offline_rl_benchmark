from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np

import pickle
from itertools import combinations



def load_policy_perf():
    # env_name: {policy_name: score}
    return pickle.load(open("d4rl_results/env_discounted_scores.pkl", "rb"))


def load_fqe_bootstrap():
    # env_name: {policy_name: np.array}
    # np.array shape is [4, 100]
    return pickle.load(open("d4rl_results/all_checkpoints/fqe_scores.pkl", "rb"))


def load_ope_scores():
    # env_name: {policy_name: np.array}
    # np.array shape is [4, 1]
    return pickle.load(open("d4rl_results/all_checkpoints/ope_scores_fulldataset.pkl", "rb"))


def solve_for_alpha(ope_scores, error_matrix_A, verbose=False):

    n = error_matrix_A.shape[0]

    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(x, error_matrix_A))
    constraints = [cp.sum(x) == 1]
    # prob = cp.Problem(objective, constraints)
    prob = cp.Problem(objective, constraints)
    if verbose:
        print('cvxpy loss: ', prob.solve())
    else:
        prob.solve()

    alpha = x.value.flatten()

    ope_scores = ope_scores.flatten()

    score = (ope_scores * alpha).sum()

    return alpha, score


def create_mse_matrix(ope_scores, ope_bootstrapped_scores):
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

    error_matrix_A = (1/num_bootstrap) * np.matmul(pre_A, pre_A.T)
    ope_mse = np.diagonal(error_matrix_A)

    return error_matrix_A, ope_mse, est_bias, est_variance


def run_experiment(policy_perfs, fqe_bootstraps, ope_scores):

    # collect data to save
    alpha_mat = {}
    score_mat = {}
    switch_ope_mat = {}
    avg_ope_mat = {}

    for env_name, policy_name_to_est in fqe_bootstraps.items():
        if env_name == 'walker2d-medium-v2':
            continue

        # print(env_name)
        alpha_mat[env_name] = {}
        score_mat[env_name] = {}
        switch_ope_mat[env_name] = {}
        avg_ope_mat[env_name] = {}

        for policy_name, policy_ope_bootstrap in policy_name_to_est.items():
            # print(policy_name)
            ope_scores_env = ope_scores[env_name][policy_name] # [:3, :]
            # ope_scores_env = np.delete(ope_scores_env, 1, axis=0)
            # policy_ope_bootstrap = policy_ope_bootstrap[:3, :]
            # policy_ope_bootstrap = np.delete(policy_ope_bootstrap, 1, axis=0)

            n_estimators =ope_scores_env.shape[0]

            # ope_scores_env = ope_scores_env * 1000  # because we normalized reward
            # policy_ope_bootstrap = policy_ope_bootstrap * 1000

            error_matrix_A, ope_mse, est_bias, est_variance = create_mse_matrix(ope_scores_env, policy_ope_bootstrap)

            alpha, score = solve_for_alpha(ope_scores_env, error_matrix_A)
            score_mat[env_name][policy_name] = score
            alpha_mat[env_name][policy_name] = alpha

            mse_smallest_idx = np.argmin(ope_mse)
            switch_ope_mat[env_name][policy_name] = ope_scores_env[mse_smallest_idx].item()
            avg_ope_mat[env_name][policy_name] = ope_scores_env.mean()

    pickle.dump(score_mat, open("d4rl_results/score_mat.pkl", "wb"))
    pickle.dump(alpha_mat, open("d4rl_results/alpha_mat.pkl", "wb"))

    # create the ope csv
    header = ['policy', 'env_name', 'true_perf'] + [f'fqe_{i}' for i in range(n_estimators)] + ['OPERA'] + ['SwitchOPE'] + ['AvgOPE']
    rows = []
    for env_name, policy_name_to_est in fqe_bootstraps.items():
        if env_name == 'walker2d-medium-v2':
            continue
        for policy_name, policy_ope_bootstrap in policy_name_to_est.items():
            ope_scores_env = ope_scores[env_name][policy_name]
            # ope_scores_env = ope_scores_env[:3, :]
            # ope_scores_env = np.delete(ope_scores_env, 1, axis=0)

            row = [policy_name, env_name] + [policy_perfs[env_name][policy_name]] + list(ope_scores_env.flatten() * 1000) + \
                  [score_mat[env_name][policy_name] * 1000] + [switch_ope_mat[env_name][policy_name] * 1000 ] + [avg_ope_mat[env_name][policy_name] * 1000]
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv('d4rl_results/ope.csv', index=False)

    # create the MSE csv
    header = ['policy', 'env_name', 'true_perf'] + [f'fqe_{i}' for i in range(n_estimators)] + ['OPERA'] + [
        'SwitchOPE'] + ['AvgOPE']
    rows = []
    for env_name, policy_name_to_est in fqe_bootstraps.items():
        if env_name == 'walker2d-medium-v2':
            continue
        for policy_name, policy_ope_bootstrap in policy_name_to_est.items():
            true_perf = policy_perfs[env_name][policy_name]
            ope_scores_env = ope_scores[env_name][policy_name]
            # ope_scores_env = ope_scores_env[:3, :]
            # ope_scores_env = np.delete(ope_scores_env, 1, axis=0)

            ope_mses = (ope_scores_env.flatten() * 1000 - true_perf) ** 2
            opera_mse = (score_mat[env_name][policy_name] * 1000 - true_perf) ** 2
            switch_mse = (switch_ope_mat[env_name][policy_name] * 1000 - true_perf) ** 2
            avg_ope_mse = (avg_ope_mat[env_name][policy_name] * 1000 - true_perf) ** 2

            row = [policy_name, env_name] + [policy_perfs[env_name][policy_name]] + list(ope_mses) + \
                  [opera_mse] + [switch_mse] + [avg_ope_mse]
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv('d4rl_results/mse.csv', index=False)

    # we shouldn't just save mean, should save everything, and find ALL env higher, not just average higher
    opera_mse = df.groupby("env_name")['OPERA'].mean()
    switch_mse = df.groupby("env_name")['SwitchOPE'].mean()
    avg_mse = df.groupby("env_name")['AvgOPE'].mean()

    return opera_mse, switch_mse, avg_mse

def filter_bad_fqes(fqe_bootstraps, filter_for_env='walker2d-medium-replay-v2'):
    # we need to identify the index of FQE that explodes on 1 policy
    # and remove that index from ALL

    fqe_idx_to_remove = []
    for policy_name, fqe_for_policy in fqe_bootstraps[filter_for_env].items():
        # fqe_for_policy: (n_estimators, n_bootstrap)
        # we remove fqe for ANY policy that blows up
        for idx in range(fqe_for_policy.shape[0]):
            if np.sum(fqe_for_policy[idx, :] > 1000) > 1:
                fqe_idx_to_remove.append(idx)

    print(fqe_idx_to_remove)

def run_search(n=4):
    n_estimators = 17
    column_indices = np.arange(n_estimators)

    # Generate all combinations of 4 columns
    combinations_n = list(combinations(column_indices, n))

    perfs = load_policy_perf()

    combo_to_mse = {}

    for combo in tqdm(combinations_n):
        fqe_bootstraps = load_fqe_bootstrap()
        scores = load_ope_scores()

        selected_fqe_boostraps = {}
        selected_scores = {}

        for env_name, policy_name_to_est in fqe_bootstraps.items():
            selected_fqe_boostraps[env_name] = {}
            selected_scores[env_name] = {}
            for policy_name, policy_ope_bootstrap in policy_name_to_est.items():
                selected_fqe_boostraps[env_name][policy_name] = policy_ope_bootstrap[combo, :]
                selected_scores[env_name][policy_name] = scores[env_name][policy_name][combo, :]

        try:
            opera_mse, switch_mse, avg_mse = run_experiment(perfs, selected_fqe_boostraps, selected_scores)
            combo_to_mse[combo] = (opera_mse, switch_mse, avg_mse)
        except:
            print("failed")

    pickle.dump(combo_to_mse, open(f"d4rl_results/combo_{n}_to_mse.pkl", "wb"))

def fqe_to_run():
    combo = (0, 8, 11, 16)
    fqe_bootstraps = load_fqe_bootstrap()
    scores = load_ope_scores()
    perfs = load_policy_perf()

    selected_fqe_boostraps = {}
    selected_scores = {}

    for env_name, policy_name_to_est in fqe_bootstraps.items():
        selected_fqe_boostraps[env_name] = {}
        selected_scores[env_name] = {}
        for policy_name, policy_ope_bootstrap in policy_name_to_est.items():
            selected_fqe_boostraps[env_name][policy_name] = policy_ope_bootstrap[combo, :]
            selected_scores[env_name][policy_name] = scores[env_name][policy_name][combo, :]

    opera_mse, switch_mse, avg_mse = run_experiment(perfs, selected_fqe_boostraps, selected_scores)

if __name__ == '__main__':
    pass

    # perfs = load_policy_perf()
    #
    # fqe_bootstraps = load_fqe_bootstrap()
    # filter_bad_fqes(fqe_bootstraps)
    # scores = load_ope_scores()
    #
    # run_experiment(perfs, fqe_bootstraps, scores)

    run_search(2)
    run_search(3)
    run_search(4)
    run_search(5)
    run_search(6)

    # fqe_to_run()