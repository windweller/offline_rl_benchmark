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


def solve_for_alpha(ope_scores, error_matrix_A):
    # ope_mse = np.outer(ope_biases, ope_biases)

    # if estimated_mse:
    # for i in range(ope_mses.shape[0]):
    #     print(ope_mse[i, i], ope_mses[i])
    #     ope_mse[i, i] = ope_mses[i]

    n = error_matrix_A.shape[0]

    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(x, error_matrix_A))
    constraints = [cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    print('cvxpy loss: ', prob.solve())

    alpha = x.value.flatten()

    score = (ope_scores * alpha).sum()

    return alpha, score


def create_mse_matrix():
    pass


def run_experiment():
    pass


if __name__ == '__main__':
    perfs = load_policy_perf()
    print(perfs)

    fqe_bootstraps = load_fqe_bootstrap()
