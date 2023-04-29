from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
    get_sepsis_subsample_copies, get_sepsis_population_full
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


def create_mse_matrix(evaluation_policy, dataset='pomdp-200',
                      ope_scorers=[], n_copies=20, sampling_procedure='bootstrap',
                      ris=False, efficient_bootstrap=False, no_clip=True):
    # This doesn't work because we need to bootstrap actual patient trajectories
    # not individual states

    datasets, full_dataset, sepsis = get_sepsis_population_full(dataset, load_first_n=n_copies)

    # orig_dataset, sepsis = get_sepsis(dataset)
    full_dataset_w_prob = convert_dataset_for_is_ope(full_dataset)
    print(len(full_dataset.episodes))

    n = len(ope_scorers)
    mat_n = n if ris is False else n * 2
    ope_mses = np.zeros(mat_n)
    ope_true_bias = np.zeros(mat_n)
    ope_true_var = np.zeros(mat_n)

    n_copies = len(datasets)
    ope_scores = np.zeros(mat_n)
    pre_A = np.zeros((mat_n, n_copies))
    ope_bootstrapped_scores = np.zeros((mat_n, n_copies))

    sub_optimal_policy = evaluation_policy

    for j, _ in enumerate(ope_scorers):

        n_bootstrap = len(datasets)
        bootstrap_ests = np.zeros(n_bootstrap)

        for i, dataset in tqdm(enumerate(datasets), total=n_copies):
            dataset_with_prob = convert_dataset_for_is_ope(dataset)
            clipped_is_score = ope_scorers[j](sub_optimal_policy, dataset_with_prob.episodes, no_clip=no_clip)
            bootstrap_ests[i] = clipped_is_score
            ope_bootstrapped_scores[j, i] = clipped_is_score

        # mean = np.mean(bootstrap_means)
        ope_score = ope_scorers[j](sub_optimal_policy, full_dataset_w_prob.episodes)
        true_bias = np.mean(bootstrap_ests - ope_score)
        true_variance = np.mean((bootstrap_ests - np.mean(bootstrap_ests)) ** 2)

        # bias = np.mean(bootstrap_means - ope_score)
        # mse = np.mean((bootstrap_means - ope_score) ** 2)

        # ope_biases[j] = bias
        ope_scores[j] = ope_score
        # ope_mses[j] = mse

        ope_bootstrapped_scores[j, :] = bootstrap_ests
        pre_A[j, :] = bootstrap_ests - ope_score
        ope_true_bias[j] = true_bias
        ope_true_var[j] = true_variance

    error_matrix_A = (1 / n_copies) * np.matmul(pre_A, pre_A.T)
    ope_mse = np.diagonal(error_matrix_A)

    return ope_scores, ope_bootstrapped_scores, ope_true_bias, ope_true_var, ope_mse, error_matrix_A


def solve_for_alpha(ope_scores, error_matrix_A):
    # ope_mse = np.outer(ope_biases, ope_biases)

    # if estimated_mse:
    #     for i in range(ope_mses.shape[0]):
    #         print(ope_mse[i, i], ope_mses[i])
    #         ope_mse[i, i] = ope_mses[i]

    n = error_matrix_A.shape[0]

    x = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(x, error_matrix_A))
    constraints = [cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    print('cvxpy loss: ', prob.solve())

    alpha = x.value.flatten()

    score = (ope_scores * alpha).sum()

    return alpha, score


def run_exp1(dataset_name='pomdp-200', n_copies=20, ris=False,
             efficient_bootstrap=False, no_clip=True):
    # this is only for the purpose of downloading policies
    datasets = get_sepsis_ensemble_datasets('pomdp-200')

    orig_dataset, sepsis = get_sepsis(dataset_name)
    policies = load_sepsis_ensemble_policies(sepsis)
    use_ris = '_ris' if ris else ''
    use_efficient = '_efficient' if efficient_bootstrap else ''

    if no_clip:
        eval_ope_names = ['is', 'wis']
    else:
        eval_ope_names = ['clipped_is', 'clipped_wis']

    del policies[1]
    policies = policies[:2]  # we take the first two policies

    print("Running ensemble performance")

    if not ris:
        alpha_mat = []
        alpha_names = ['alpha_clipped_is', 'alpha_clipped_weighted_is']

        ensemble_scores = []
        opes_mat = []
        ope_names = ['true_perf'] + eval_ope_names + ['ope_ensemble']

        sampled_ope_mse = []
        sampled_ope_mse_name = [f'sampled_{n}_mse' for n in eval_ope_names]

        sampled_ope_biases = []
        sampled_ope_biases_name = [f'sampled_{n}_bias' for n in eval_ope_names]

        sampled_ope_variances = []
        sampled_ope_variances_name = [f'sampled_{n}_variance' for n in eval_ope_names]

        # real_ope_biases = []
        # real_ope_biases_name = ['real_clipped_is_bias', 'real_clipped_wis_bias']

        # for this particular dataset, not the true MSE
        mse_mat = []
        mse_names = [f'mse_{n}' for n in eval_ope_names] + ['ope_ensemble_mse']

    else:
        raise Exception("Not implemented yet")
        alpha_mat = []
        alpha_names = ['alpha_clipped_is', 'alpha_clipped_weighted_is', 'alpha_ris_clipped_is',
                       'alpha_ris_clipped_weighted_is']

        ensemble_scores = []
        opes_mat = []
        ope_names = ['true_perf', 'clipped_is', 'clipped_weighted_is', 'ris_clipped_is', 'ris_clipped_weighted_is',
                     'ope_ensemble']

        sampled_ope_biases = []
        sampled_ope_biases_name = ['sampled_clipped_is_bias', 'sampled_clipped_wis_bias', 'sampled_ris_clipped_is_bias',
                                   'sampled_ris_clipped_wis_bias']

        # real_ope_biases = []
        # real_ope_biases_name = ['real_clipped_is_bias', 'real_clipped_wis_bias', 'real_ris_clipped_is_bias', 'real_ris_clipped_wis_bias']

        mse_mat = []
        mse_names = ['mse_clipped_is', 'mse_clipped_weighted_is', 'mse_ris_clipped_is', 'mse_ris_clipped_weighted_is',
                     'mse_ensemble_score']

    for i in tqdm(range(len(policies))):
        # we can't evaluate policies[1] because it's the sampling policy
        # but we can evaluate the rest
        ope_row = []

        true_perf = POLICY_TRUE_MEAN_PERF[i]
        ope_row.append(true_perf)

        ope_scores, ope_bootstrapped_scores, est_bias, est_variance, ope_mse, error_matrix_A = create_mse_matrix(
            policies[i], ope_scorers=[importance_sampling_scorer, wis_scorer],
            dataset=dataset_name, n_copies=n_copies, ris=ris, efficient_bootstrap=efficient_bootstrap, no_clip=no_clip)

        np.savez(f"sepsis_results/policy_{i}_{dataset_name}_{n_copies}{use_ris}{use_efficient}_ope_true_copies_scores.npz",
                 ope_bootstrapped_scores=ope_bootstrapped_scores)

        alphas, score = solve_for_alpha(ope_scores, error_matrix_A)
        alpha_mat.append(alphas.flatten().tolist())

        ope_row.extend(ope_scores.tolist())
        # print(ope_scores)
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
    true_ope_biases_mat = pd.DataFrame(sampled_ope_biases, columns=sampled_ope_biases_name)
    true_ope_variances_mat = pd.DataFrame(sampled_ope_variances, columns=sampled_ope_variances_name)
    mse_mat = pd.DataFrame(mse_mat, columns=mse_names)

    opes_mat.to_csv(f'sepsis_results/{dataset_name}_{n_copies}{use_ris}{use_efficient}_true_opes_mat.csv')
    alpha_mat.to_csv(f'sepsis_results/{dataset_name}_{n_copies}{use_ris}{use_efficient}_true_alpha_mat.csv')
    true_ope_biases_mat.to_csv(
        f'sepsis_results/{dataset_name}_{n_copies}{use_ris}{use_efficient}_true_ope_biases_mat.csv')
    true_ope_variances_mat.to_csv(f'sepsis_results/{dataset_name}_{n_copies}_true_ope_variances_mat.csv')
    mse_mat.to_csv(f'sepsis_results/{dataset_name}_{n_copies}{use_ris}{use_efficient}_true_mse_mat.csv')


if __name__ == '__main__':
    ...
    # need to call this to download the policies
    run_exp1('pomdp-200', n_copies=50, efficient_bootstrap=False, no_clip=True)  # , efficient_bootstrap=True
    # run_exp1('pomdp-1000', n_copies=5, efficient_bootstrap=False, no_clip=True) # , efficient_bootstrap=True

    # run_exp1('pomdp-1000', n_copies=100)
    # run_exp2('pomdp-1000', n_copies=20)

    # run_exp1('pomdp-200', n_copies=20, ris=True)
    # run_exp1('pomdp-1000', n_copies=20, ris=True)

    # run_exp2_subsample('pomdp-200', n_copies=20)
    # run_exp2_subsample('pomdp-1000', n_copies=20)

    # Try RIS on this dataset!!! And see if it works

    # dataset_name = 'pomdp-200'
    # dataset, sepsis = get_sepsis(dataset_name)
    # policies = load_sepsis_ensemble_policies(sepsis)

    # bc = DiscreteBC(use_gpu=False)
    # bc.build_with_dataset(dataset)
    # ris = RegressionIS(bc)
    # ris.fit(dataset, n_epochs=25)
    # prob_dataset = ris.estimate_bh_probabilities(dataset)
    #
    # dataset_with_prob = convert_dataset_for_is_ope(prob_dataset)
    #
    # clipped_is_score = importance_sampling_scorer(policies[0], dataset_with_prob.episodes)
    # weighted_is_score = wis_scorer(policies[0], dataset_with_prob.episodes)
    #
    # print(clipped_is_score)
    # print(weighted_is_score)
