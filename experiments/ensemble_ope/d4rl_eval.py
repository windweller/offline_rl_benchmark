from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np

import pickle


def load_policy_perf():
    # env_name: {policy_name: score}
    return pickle.load(open("d4rl_results/env_scores.pkl", "rb"))


def load_fqe_bootstrap():
    # env_name: {policy_name: np.array}
    # np.array shape is [4, 100]
    return pickle.load(open("d4rl_results/fqe_scores.pkl", "rb"))


def load_ope_scores():
    # env_name: {policy_name: np.array}
    # np.array shape is [4, 1]
    return pickle.load(open("d4rl_results/ope_scores.pkl", "rb"))


def solve_for_alpha(ope_scores, error_matrix_A):

    n = error_matrix_A.shape[0]

    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(x, error_matrix_A))
    constraints = [cp.sum(x) == 1]
    # prob = cp.Problem(objective, constraints)
    prob = cp.Problem(objective, constraints)
    print('cvxpy loss: ', prob.solve())

    alpha = x.value.flatten()

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
        print(env_name)
        alpha_mat[env_name] = {}
        score_mat[env_name] = {}
        switch_ope_mat[env_name] = {}
        avg_ope_mat[env_name] = {}

        for policy_name, policy_ope_bootstrap in tqdm(policy_name_to_est.items()):
            print(policy_name)
            ope_scores_env = ope_scores[env_name][policy_name]
            n_estimators =ope_scores_env.shape[0]

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
        for policy_name, policy_ope_bootstrap in policy_name_to_est.items():
            ope_scores_env = ope_scores[env_name][policy_name]
            row = [policy_name, env_name] + [policy_perfs[env_name][policy_name]] + list(ope_scores_env.flatten()) + \
                  [score_mat[env_name][policy_name]] + [switch_ope_mat[env_name][policy_name]] + [avg_ope_mat[env_name][policy_name]]
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv('d4rl_results/ope.csv', index=False)


if __name__ == '__main__':
    perfs = load_policy_perf()
    print(perfs)

    fqe_bootstraps = load_fqe_bootstrap()
    scores = load_ope_scores()

    run_experiment(perfs, fqe_bootstraps, scores)
