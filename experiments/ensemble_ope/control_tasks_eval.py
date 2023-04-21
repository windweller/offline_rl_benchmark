import os
import argparse
import numpy as np
from tqdm import tqdm

import pandas as pd

import gym
import d3rlpy
from d3rlpy.algos import DQN, DiscreteBCQ, DiscreteCQL
from d3rlpy.ope import DiscreteFQE
from d3rlpy.dataset import MDPDataset

from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import evaluate_on_environment, initial_state_value_estimation_scorer

import cvxpy as cp

from offline_rl.envs.dataset import sample_bootstrap


def load_in_control_policies(poilcy_dir: str, env: gym.Env, dataset: MDPDataset):
    """
    We assume policy_dir contains policy.pt files
    We would load in and evaluate their online performance
    :param poilcy_dir:
    :return:
    """
    evaluate_scorer = evaluate_on_environment(env)

    policies_w_perf = []
    for f in os.listdir(poilcy_dir):
        if f.endswith('.pt'):
            bcq = DiscreteBCQ()
            bcq.build_with_dataset(dataset)
            bcq.load_model(os.path.join(poilcy_dir, f))
            policies_w_perf.append((bcq, evaluate_scorer(bcq)))

    policies_w_perf = sorted(policies_w_perf, key=lambda x: x[1], reverse=False)
    policies = [p[0] for p in policies_w_perf]
    policy_perfs = [p[1] for p in policies_w_perf]

    return policies, policy_perfs


fqe_configs = [1, 3, 5, 7, 9]

def create_mse_matrix(policy, dataset_name, save_dir):
    print("Running ensemble performance")

    assert dataset_name in ['cartpole-replay', 'cartpole-random']
    dataset, env = d3rlpy.datasets.get_dataset(dataset_name)

    _, eval_episodes = train_test_split(dataset, test_size=0.5, random_state=42)

    ope_biases = np.zeros(len(fqe_configs))
    ope_mses = np.zeros(len(fqe_configs))
    ope_scores = np.zeros(len(fqe_configs))
    ope_bootstrapped_scores = np.zeros((len(fqe_configs), 10))

    print("Running ensemble performance")

    # this step is not even necessary...
    bootstrapped_datasets, scale_ratio = sample_bootstrap(eval_episodes, env, 10)

    # we will just train a bunch of FQEs
    for j, fqe_epoch in enumerate(fqe_configs):
        bootstrap_means = np.zeros(len(bootstrapped_datasets))

        for i, bootstrapped_dataset in enumerate(bootstrapped_datasets):
            fqe = DiscreteFQE(algo=policy)
            fqe.build_with_dataset(bootstrapped_dataset)
            # we can do an honest estimation here
            fqe.fit(bootstrapped_dataset.episodes,
                    n_epochs=fqe_epoch,
                    experiment_name=f'{dataset_name}_fqe_eps_{fqe_epoch}',
                    save_interval=10)
            fqe_score = initial_state_value_estimation_scorer(fqe, bootstrapped_dataset.episodes)
            bootstrap_means[i] = fqe_score

        fqe = DiscreteFQE(algo=policy)
        fqe.build_with_dataset(dataset)
        # we can do an honest estimation here
        fqe.fit(eval_episodes,
                n_epochs=fqe_epoch,
                experiment_name=f'{dataset_name}_fqe_eps_{fqe_epoch}',
                save_interval=10)
        ope_score = initial_state_value_estimation_scorer(fqe, eval_episodes)
        bias = np.mean(bootstrap_means - ope_score)
        mse = scale_ratio * np.mean((bootstrap_means - ope_score) ** 2)

        ope_biases[j] = bias
        ope_scores[j] = ope_score
        ope_mses[j] = mse

    return ope_biases, ope_mses, ope_scores, ope_bootstrapped_scores


def solve_for_alpha(ope_biases: np.array, ope_mses: np.array, ope_scores: np.array):
    ope_mse = np.outer(ope_biases, ope_biases)

    # if estimated_mse:
    for i in range(ope_mses.shape[0]):
        print(ope_mse[i, i], ope_mses[i])
        ope_mse[i, i] = ope_mses[i]

    n = ope_biases.shape[0]

    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(x, ope_mse))
    constraints = [cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    print('cvxpy loss: ', prob.solve())

    alpha = x.value.flatten()

    score = (ope_scores * alpha).sum()

    return alpha, score


def run_experiment(policy_dir: str, dataset_name: str, save_dir: str):
    print("Running ensemble performance")

    assert dataset_name in ['cartpole-replay', 'cartpole-random']
    dataset, env = d3rlpy.datasets.get_dataset(dataset_name)

    alpha_mat = []
    alpha_names = [f'alpha_fqe{i}' for i in range(len(fqe_configs))]

    ensemble_scores = []
    opes_mat = []
    ope_names = ['true_perf'] + [f'fqe{i}' for i in range(len(fqe_configs))] + ['ope_ensemble']

    sampled_ope_biases = []
    sampled_ope_biases_name = [f'sampled_fqe_{i}_bias' for i in range(len(fqe_configs))]

    real_ope_biases = []
    real_ope_biases_name = [f'real_fqe_{i}_bias' for i in range(len(fqe_configs))]

    mse_mat = []
    mse_names = [f'mse_fqe{i}' for i in range(len(fqe_configs))] + ['mse_ope_ensemble']

    policies, policy_perfs = load_in_control_policies(policy_dir, env, dataset)
    n_copies = 10

    for i in tqdm(range(len(policies))):
        # we can't evaluate policies[1] because it's the sampling policy
        # but we can evaluate the rest
        ope_row = []

        true_perf = policy_perfs[i]
        ope_row.append(true_perf)

        ope_biases, ope_mses, ope_scores, ope_bootstrapped_scores = create_mse_matrix(policies[i], dataset_name,
                                                                                      save_dir)

        np.savez(f"control_tasks_results/policy_{i}_{dataset_name}_{n_copies}_ope_subsampled_scores.npz",
                 ope_bootstrapped_scores=ope_bootstrapped_scores)

        alphas, score = solve_for_alpha(ope_biases, ope_mses, ope_scores)
        alpha_mat.append(alphas.flatten().tolist())

        ope_row.extend(ope_scores.tolist())
        ope_row.append(score)

        opes_mat.append(ope_row)
        eval_scores = ope_scores.tolist() + [score]
        mse_mat.append(((np.array(eval_scores) - true_perf) ** 2).tolist())

        sampled_ope_biases.append(ope_biases.flatten().tolist())

        true_bias = ope_scores - true_perf
        real_ope_biases.append(true_bias.tolist())

    opes_mat = pd.DataFrame(opes_mat, columns=ope_names)
    alpha_mat = pd.DataFrame(alpha_mat, columns=alpha_names)
    sampled_ope_biases_mat = pd.DataFrame(sampled_ope_biases, columns=sampled_ope_biases_name)
    real_ope_biases_mat = pd.DataFrame(real_ope_biases, columns=real_ope_biases_name)
    mse_mat = pd.DataFrame(mse_mat, columns=mse_names)

    opes_mat.to_csv(f'results/{dataset_name}_{n_copies}_opes_mat.csv')
    alpha_mat.to_csv(f'results/{dataset_name}_{n_copies}_alpha_mat.csv')
    sampled_ope_biases_mat.to_csv(f'results/{dataset_name}_{n_copies}_sampled_ope_biases_mat.csv')
    real_ope_biases_mat.to_csv(f'results/{dataset_name}_{n_copies}_real_ope_biases_mat.csv')
    mse_mat.to_csv(f'results/{dataset_name}_{n_copies}_mse_mat.csv')


if __name__ == '__main__':
    run_experiment("model_logs/cartpole_policies", "cartpole-replay", "model_logs")
