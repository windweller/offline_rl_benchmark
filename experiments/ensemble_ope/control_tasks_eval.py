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

def get_dataset(dataset_name, save_dir):
    assert dataset_name in ['cartpole-replay', 'cartpole-random', 'acrobot-replay', 'acrobot-random']

    if 'acrobot' not in dataset_name:
        dataset, env = d3rlpy.datasets.get_dataset(dataset_name)
    else:
        # we load in differently
        if 'replay' in dataset_name:
            data_path = os.path.join(save_dir, 'acrobot_v1_replay_dataset.h5')
        else:
            data_path = os.path.join(save_dir, 'acrobot_v1_random_dataset.h5')
        dataset = MDPDataset.load(data_path)
        env = gym.make('Acrobot-v1')

    return dataset, env

def load_in_control_policies(poilcy_dir: str, dataset_name: str, save_dir: str):
    """
    We assume policy_dir contains policy.pt files
    We would load in and evaluate their online performance
    :param poilcy_dir:
    :return:
    """
    dataset, env = get_dataset(dataset_name, save_dir)
    evaluate_scorer = evaluate_on_environment(env)

    policies_w_perf = []
    for f in os.listdir(poilcy_dir):
        if f.endswith('.pt'):
            if 'acrobot' not in dataset_name:
                bcq = DiscreteBCQ()
            else:
                bcq = DQN()
            bcq.build_with_dataset(dataset)
            bcq.load_model(os.path.join(poilcy_dir, f))
            policies_w_perf.append((bcq, evaluate_scorer(bcq)))

    policies_w_perf = sorted(policies_w_perf, key=lambda x: x[1], reverse=False)
    policies = [p[0] for p in policies_w_perf]
    policy_perfs = [p[1] for p in policies_w_perf]

    return policies, policy_perfs


# fqe_configs = [1, 3, 5, 7, 9]
fqe_configs = [(2, [32, 32]),
               (2, [64, 64]),
               (5, [32, 32]),
               (5, [64, 64]),
               (8, [32, 32]),
               (8, [64, 64])]

def create_mse_matrix(policy, dataset_name, save_dir, num_bootstrap=10, k_prop=0.15):
    print("Running ensemble performance")

    assert dataset_name in ['cartpole-replay', 'cartpole-random', 'acrobot-replay', 'acrobot-random']
    # dataset, env = d3rlpy.datasets.get_dataset(dataset_name)
    dataset, env = get_dataset(dataset_name, save_dir)

    _, eval_episodes = train_test_split(dataset, test_size=0.5, random_state=42)

    ope_scores = np.zeros(len(fqe_configs))
    ope_est_bias = np.zeros(len(fqe_configs))
    ope_est_var = np.zeros(len(fqe_configs))
    # P x B
    pre_A = np.zeros((len(fqe_configs), num_bootstrap))
    ope_bootstrapped_scores = np.zeros((len(fqe_configs), num_bootstrap))

    print("Running ensemble performance")

    bootstrapped_datasets, scale_ratio = sample_bootstrap(eval_episodes, env, num_bootstrap, k_prop=k_prop)

    # we will just train a bunch of FQEs
    for j, (fqe_epoch, fqe_hid_dims) in enumerate(fqe_configs):
        bootstrap_ests = np.zeros(len(bootstrapped_datasets))
        encoder = d3rlpy.models.encoders.VectorEncoderFactory(fqe_hid_dims)

        for i, bootstrapped_dataset in enumerate(bootstrapped_datasets):

            fqe = DiscreteFQE(algo=policy, encoder_factory=encoder)
            fqe.build_with_dataset(bootstrapped_dataset)
            # we can do an honest estimation here
            fqe.fit(bootstrapped_dataset.episodes,
                    n_epochs=fqe_epoch,
                    experiment_name=f'{dataset_name}_fqe_eps_{fqe_epoch}_hid_{"-".join([str(h) for h in fqe_hid_dims])}',
                    save_interval=10)
            fqe_score = initial_state_value_estimation_scorer(fqe, bootstrapped_dataset.episodes)
            bootstrap_ests[i] = fqe_score

        fqe = DiscreteFQE(algo=policy, encoder_factory=encoder)
        fqe.build_with_dataset(dataset)
        # we can do an honest estimation here
        fqe.fit(eval_episodes,
                n_epochs=fqe_epoch,
                experiment_name=f'{dataset_name}_fqe_eps_{fqe_epoch}_hid_{"-".join([str(h) for h in fqe_hid_dims])}',
                save_interval=10)
        ope_score = initial_state_value_estimation_scorer(fqe, eval_episodes)

        # we have to compare these with the true bias and true variance
        est_bias = np.mean(bootstrap_ests - ope_score)
        est_variance = np.mean((bootstrap_ests - np.mean(bootstrap_ests)) ** 2)

        # j-th estimator
        ope_bootstrapped_scores[j, :] = bootstrap_ests
        pre_A[j, :] = bootstrap_ests - ope_score
        ope_scores[j] = ope_score
        ope_est_bias[j] = est_bias
        ope_est_var[j] = est_variance

    # (P x B) x (B x P) = P x P
    error_matrix_A = scale_ratio * (1 / num_bootstrap) * np.matmul(pre_A, pre_A.T)
    ope_mse = np.trace(error_matrix_A)

    return ope_scores, ope_bootstrapped_scores, ope_est_bias, ope_est_var,  ope_mse, error_matrix_A


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


def run_experiment(policy_dir: str, dataset_name: str, save_dir: str):
    print("Running ensemble performance")

    # assert dataset_name in ['cartpole-replay', 'cartpole-random', 'acrobot-replay', 'acrobot-random']
    # dataset, env = d3rlpy.datasets.get_dataset(dataset_name)
    dataset, env = get_dataset(dataset_name, save_dir)

    alpha_mat = []
    alpha_names = [f'alpha_fqe{i}' for i in range(len(fqe_configs))]

    opes_mat = []
    ope_names = ['true_perf'] + [f'fqe{i}' for i in range(len(fqe_configs))] + ['ope_ensemble']

    sampled_ope_mse = []
    sampled_ope_mse_name = [f'sampled_fqe_{i}_bias' for i in range(len(fqe_configs))]

    sampled_ope_biases = []
    sampled_ope_biases_name = [f'sampled_fqe_{i}_bias' for i in range(len(fqe_configs))]

    sampled_ope_variances = []
    sampled_ope_variances_name = [f'sampled_fqe_{i}_variance' for i in range(len(fqe_configs))]

    # real_ope_biases = []
    # real_ope_biases_name = [f'real_fqe_{i}_bias' for i in range(len(fqe_configs))]

    mse_mat = []
    mse_names = [f'mse_fqe{i}' for i in range(len(fqe_configs))] + ['mse_ope_ensemble']

    policies, policy_perfs = load_in_control_policies(policy_dir, dataset_name, save_dir)
    n_copies = 20

    for i in tqdm(range(len(policies))):
        # we can't evaluate policies[1] because it's the sampling policy
        # but we can evaluate the rest
        ope_row = []

        true_perf = policy_perfs[i]
        ope_row.append(true_perf)

        ope_scores, ope_bootstrapped_scores, est_bias, est_variance,  ope_mse, error_matrix_A = create_mse_matrix(policies[i], dataset_name,
                                                                                      save_dir, num_bootstrap=n_copies)

        np.savez(f"control_tasks_results/policy_{i}_{dataset_name}_{n_copies}_ope_subsampled_scores.npz",
                 ope_bootstrapped_scores=ope_bootstrapped_scores, ope_scores=ope_scores)

        alphas, score = solve_for_alpha(ope_scores, error_matrix_A)
        alpha_mat.append(alphas.flatten().tolist())

        ope_row.extend(ope_scores.tolist())
        ope_row.append(score)

        opes_mat.append(ope_row)
        eval_scores = ope_scores.tolist() + [score]
        mse_mat.append(((np.array(eval_scores) - true_perf) ** 2).tolist())

        sampled_ope_biases.append(est_bias.flatten().tolist())
        sampled_ope_variances.append(est_variance.flatten().tolist())
        sampled_ope_mse.append(ope_mse.flatten().tolist())

        # true_bias = ope_scores - true_perf
        # real_ope_biases.append(true_bias.tolist())

    opes_mat = pd.DataFrame(opes_mat, columns=ope_names)
    alpha_mat = pd.DataFrame(alpha_mat, columns=alpha_names)
    sampled_ope_biases_mat = pd.DataFrame(sampled_ope_biases, columns=sampled_ope_biases_name)
    sampled_ope_variances_mat = pd.DataFrame(sampled_ope_variances, columns=sampled_ope_variances_name)
    # real_ope_biases_mat = pd.DataFrame(real_ope_biases, columns=real_ope_biases_name)
    mse_mat = pd.DataFrame(mse_mat, columns=mse_names)
    mse_est = pd.DataFrame(sampled_ope_mse, columns=sampled_ope_mse_name)

    opes_mat.to_csv(f'results/{dataset_name}_{n_copies}_opes_mat.csv')
    alpha_mat.to_csv(f'results/{dataset_name}_{n_copies}_alpha_mat.csv')
    sampled_ope_biases_mat.to_csv(f'results/{dataset_name}_{n_copies}_sampled_ope_biases_mat.csv')
    sampled_ope_variances_mat.to_csv(f'results/{dataset_name}_{n_copies}_sampled_ope_variances_mat.csv')
    # real_ope_biases_mat.to_csv(f'results/{dataset_name}_{n_copies}_real_ope_biases_mat.csv')
    mse_mat.to_csv(f'results/{dataset_name}_{n_copies}_mse_mat.csv')
    mse_est.to_csv(f'results/{dataset_name}_{n_copies}_mse_est.csv')


if __name__ == '__main__':
    # add argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cartpole-replay')
    parser.add_argument('--save_dir', type=str, default='model_logs')
    parser.add_argument('--policy_dir', type=str, default='model_logs/cartpole_policies')
    args = parser.parse_args()

    run_experiment(args.policy_dir, args.dataset, args.save_dir)

    # run_experiment("model_logs/cartpole_policies", "cartpole-replay", "model_logs")
    # run_experiment("model_logs/cartpole_policies", "cartpole-random", "model_logs")
    # run_experiment("model_logs/acrobot_policies", "acrobot-replay", "model_logs")