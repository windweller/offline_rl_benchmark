from offline_rl.envs.datasets import get_sepsis_ensemble_exclude_one, get_sepsis_ensemble_full, get_sepsis, \
    get_sepsis_ensemble_datasets, combine_sepsis_ensemble_datasets
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF

from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment
from offline_rl.envs.dataset import convert_dataset_for_is_ope
from sklearn.model_selection import train_test_split

from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer

from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
    cwpdis_scorer, pdis_scorer

from tqdm import tqdm
import pandas as pd

import numpy as np

from offline_rl.envs.dataset import ProbabilityMDPDataset
from offline_rl.algs.policy_evaluation_wrappers import DiscreteProbabilisticPolicyProtocol


# TODO:
# Bug 2: behavior probability sometimes is 0, which is not possible...(because we choose action chosen by behavior policy)

def create_bias_matrix(dataset='pomdp-100'):
    # rows: each behavior policy
    # columns: each ope - true performance
    datasets, sepsis = get_sepsis_ensemble_exclude_one(dataset)
    # dataset, sepsis = get_sepsis('pomdp-5000')
    policies = load_sepsis_ensemble_policies(sepsis)

    bias_mat = []
    opes_mat = []
    ope_names = ['true_perf', 'clipped_is', 'weighted_is', 'cwpdis', 'pdis']

    for i in tqdm(range(len(policies))):
        bias_row = []
        ope_row = []

        true_perf = POLICY_TRUE_MEAN_PERF[i]
        # dataset_with_prob = convert_dataset_for_is_ope(dataset)
        dataset_with_prob = convert_dataset_for_is_ope(datasets[i])
        bias_row.append(true_perf)
        ope_row.append(true_perf)

        clipped_is_score = importance_sampling_scorer(policies[i], dataset_with_prob.episodes)
        bias_row.append(clipped_is_score - true_perf)
        ope_row.append(clipped_is_score)

        weighted_is_score = wis_scorer(policies[i], dataset_with_prob.episodes)
        bias_row.append(weighted_is_score - true_perf)
        ope_row.append(weighted_is_score)

        cwpdis_score = cwpdis_scorer(policies[i], dataset_with_prob.episodes)
        bias_row.append(cwpdis_score - true_perf)
        ope_row.append(cwpdis_score)

        pdis_score = pdis_scorer(policies[i], dataset_with_prob.episodes)
        bias_row.append(pdis_score - true_perf)
        ope_row.append(pdis_score)

        bias_mat.append(bias_row)
        opes_mat.append(ope_row)

        # RIS

        # fqe = DiscreteFQE(algo=policies[i])
        # fqe.build_with_dataset(datasets[i])
        #
        # train_episodes, test_episodes = train_test_split(datasets[i], test_size=0.5, random_state=3)
        # metrics = fqe.fit(train_episodes, n_epochs=2,
        #                   scorers={
        #                       'init_value': initial_state_value_estimation_scorer
        #                   })
        # # print(metrics)
        # score = initial_state_value_estimation_scorer(fqe, test_episodes)
        # print(score)
        # break

    # build pandas dataframe
    bias_mat = pd.DataFrame(bias_mat, columns=ope_names)
    opes_mat = pd.DataFrame(opes_mat, columns=ope_names)

    return bias_mat, opes_mat


def compute_empirical_kl(eval_policy: DiscreteProbabilisticPolicyProtocol,
                         behavior_policy: DiscreteProbabilisticPolicyProtocol, dataset: ProbabilityMDPDataset):
    # Note: dataset should be sampled from behavior_policy
    # range: between 0 and infinity

    # importance-sampling reweighted KL
    # logp of the evaluation policy and logp of the behavior policy

    # approx_kl = (logp_old - logp).mean().item()

    # emp_kl_over_all_traj = []
    # this is not efficient for large dataset
    eval_probs = eval_policy.predict_action_probabilities(dataset.observations)
    eval_probs = [action_prob[np.argmax(dataset.actions[i, :])] for i, action_prob in enumerate(eval_probs)]
    logp_eval = np.log(eval_probs)

    beh_p = [action_prob[np.argmax(dataset.actions[i, :])] for i, action_prob in enumerate(dataset.action_probabilities)]
    logp = np.log(beh_p)

    # KL should always be > 0, I don't know what happened, but yeah,
    return np.abs(np.mean(logp_eval - logp))


def compute_ensemble_score(env_name='pomdp-100'):
    # we currently do this by witholding one behavior poilcy (no actual training)
    # 1. get bias matrix; 2. get ope matrix (we have both already)
    # 3. get KL vector (we can get them now)
    # compute! Store result!

    bias_mat, ope_mat = create_bias_matrix(env_name)

    bias_mat = bias_mat.drop(columns=['true_perf'])
    ope_mat = ope_mat.drop(columns=['true_perf'])

    datasets = get_sepsis_ensemble_datasets(env_name)
    dataset, sepsis = combine_sepsis_ensemble_datasets(env_name, [datasets[0]])
    policies = load_sepsis_ensemble_policies(sepsis)

    scores_for_each_policy = []
    for i in range(5):
        # evaluating for policy i
        bias_mat_i = bias_mat.drop([i]).to_numpy()
        opes_mat_i = ope_mat.drop([i]).to_numpy()

        kl_vec = [] # 4
        for j in range(5):
            if i == j:
                continue
            dataset, _ = combine_sepsis_ensemble_datasets(env_name, [datasets[j]])
            dataset_with_prob = convert_dataset_for_is_ope(dataset)
            kl = compute_empirical_kl(policies[i], policies[j], dataset_with_prob)
            kl_vec.append(kl)

        kl_vec = np.array(kl_vec)

        # compute score
        score = np.matmul(1/kl_vec.reshape(1, len(kl_vec)), (1 / bias_mat_i) * opes_mat_i).sum()

        scores_for_each_policy.append(score)

    return scores_for_each_policy

if __name__ == '__main__':
    ...
    # bias_mat, ope_row = create_bias_matrix()
    # bias_mat.to_csv('bias_mat.csv', index=False)
    # ope_row.to_csv('ope_row.csv', index=False)

    # ====== IS test ======

    # dataset, sepsis = get_sepsis_ensemble_full('pomdp-100')
    # policies = get_sepsis_ensemble_policies(sepsis)
    #
    # dataset_with_prob = convert_dataset_for_is_ope(dataset)

    # TODO: remaining bugs:
    # TODO: 2. behavior probability is 0, which is not possible (now each trajectory is of dynamic length)
    # TODO: -- this fix is weird, we set terminal_idx to be last position - 2

    # print(dataset_with_prob.actions)
    # print(dataset_with_prob.actions.shape)
    #
    # batch = TransitionMiniBatch(dataset_with_prob.episodes[0].transitions, 1)
    #
    # print(batch.actions.shape)
    # print(batch.rewards.shape)
    # print(batch.terminals.shape)

    # print(np.unique(dataset_with_prob.episodes[0].observations[:, -1]))
    # print(len(dataset_with_prob.episodes[0]))
    #
    # print(dataset_with_prob.episodes[0].observations)

    # for batch in _make_batches(dataset_with_prob.episodes[0], 50, 1):
    #     print(batch.observations.shape)
    #     print(batch.observations)
    #     print(batch.next_observations)
    # print(np.unique(batch.observations[:, -1]))
    #
    # clipped_is_score = importance_sampling_scorer(policies[0], dataset_with_prob.episodes)
    # print(clipped_is_score)

    # ===== Empirical KL checkup =====

    # datasets = get_sepsis_ensemble_datasets('pomdp-100')
    # dataset, sepsis = combine_sepsis_ensemble_datasets('pomdp-100', [datasets[0]])
    # policies = get_sepsis_ensemble_policies(sepsis)
    #
    # dataset_with_prob = convert_dataset_for_is_ope(dataset)
    #
    # # verify empirical KL
    # emp_kl = compute_empirical_kl(policies[0], policies[0], dataset_with_prob)
    # print(emp_kl)
    # emp_kl = compute_empirical_kl(policies[1], policies[0], dataset_with_prob)
    # print(emp_kl)
    # emp_kl = compute_empirical_kl(policies[2], policies[0], dataset_with_prob)
    # print(emp_kl)
    # emp_kl = compute_empirical_kl(policies[3], policies[0], dataset_with_prob)
    # print(emp_kl)
    # emp_kl = compute_empirical_kl(policies[4], policies[0], dataset_with_prob)
    # print(emp_kl)

    # ===== Ensemble score =====
    # scores = compute_ensemble_score()
    # print(scores)
    #
    scores = compute_ensemble_score('pomdp-200')
    print(scores)

    # bias_mat, ope_row = create_bias_matrix('pomdp-200')
    # bias_mat.to_csv('bias_mat.csv', index=False)
    # ope_row.to_csv('ope_row.csv', index=False)
    # 
    # bias_mat, ope_row = create_bias_matrix('pomdp-500')
    # bias_mat.to_csv('bias_mat.csv', index=False)
    # ope_row.to_csv('ope_row.csv', index=False)