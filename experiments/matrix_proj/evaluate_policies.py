import os
import csv
import json
from tqdm import tqdm

import d3rlpy

from offline_rl.envs.datasets import get_sepsis
from offline_rl.algs.policy_evaluation_wrappers import DiscreteCQLEvaluationWrapper, DiscreteBCEvaluationWrapper, \
                                                       DiscreteBCQEvaluationWrapper
from offline_rl.algs.policy_evaluation_wrappers import DiscreteCQLTorchWrapper, DiscreteBCTorchWrapper, \
                                                        DiscreteBCQTorchWrapper
from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment
from offline_rl.envs.dataset import convert_dataset_for_is_ope

from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
            cwpdis_scorer, pdis_scorer
from offline_rl.opes.regression_importance_sampling import RegressionIS

from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer

from offline_rl.opes.dice_rl.estimators.neural_dice import NeuralDice
from offline_rl.opes.dice_rl.networks.value_network import ValueNetwork, ValueNetworkWithAction
from offline_rl.opes.dice_rl.data.dataset import InitialStepRandomIterator, StepNextStepRoundIterator

# We evaluate on OPE estimators

def evaluate_policy(original_policy, eval_wrapped_policy, eval_torch_policy, dataset):
    # policy_name: we need to be able to look up true perf
    #               should look like "mdp_200_bc_hcomb_0"
    evaluation_policy = eval_wrapped_policy
    dataset_with_prob = convert_dataset_for_is_ope(dataset)

    # === IS ===
    clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
    weighted_is_score = wis_scorer(evaluation_policy, dataset_with_prob.episodes)
    cwpdis_score = cwpdis_scorer(evaluation_policy, dataset_with_prob.episodes)
    pdis_score = pdis_scorer(evaluation_policy,dataset_with_prob.episodes)

    # === RIS ===

    bc = d3rlpy.algos.DiscreteBC(use_gpu=False)
    bc.build_with_dataset(dataset)
    ris = RegressionIS(bc)
    ris.fit(dataset, n_epochs=5)
    prob_dataset = ris.estimate_bh_probabilities(dataset)

    dataset_with_prob = convert_dataset_for_is_ope(prob_dataset)

    ris_clipped_is_score = importance_sampling_scorer(evaluation_policy, dataset_with_prob.episodes)
    ris_weighted_is_score = wis_scorer(evaluation_policy, dataset_with_prob.episodes)
    ris_cwpdis_score = cwpdis_scorer(evaluation_policy, dataset_with_prob.episodes)
    ris_pdis_score = pdis_scorer(evaluation_policy, dataset_with_prob.episodes)

    # === FQE ===

    fqe = d3rlpy.ope.DiscreteFQE(algo=original_policy)
    fqe.build_with_dataset(dataset)

    fqe_metrics = fqe.fit(dataset, n_epochs=5,
                      eval_episodes=dataset.episodes,
                      scorers={
                          'init_value': initial_state_value_estimation_scorer
                      })
    fqe_score = fqe_metrics[0][1]['init_value']

    # === DICE RL ===
    init_iterator = InitialStepRandomIterator(dataset, 32)
    iterator = StepNextStepRoundIterator(dataset, 32)

    solve_for_state_action_ratio = True
    zeta_pos = True
    num_steps = 5

    nu_encoder = ValueNetworkWithAction(bc.observation_shape[0] + 1, [64, 64], output_size=1,
                                        zeta_pos=zeta_pos)
    zeta_encoder = ValueNetworkWithAction(bc.observation_shape[0] + 1, [64, 64], output_size=1,
                                          zeta_pos=zeta_pos)

    estimator = NeuralDice(nu_encoder, zeta_encoder, nu_learning_rate=1e-3, zeta_learning_rate=1e-3,
                           gamma=0.99, has_log_probability=True,
                           solve_for_state_action_ratio=solve_for_state_action_ratio,
                           primal_regularizer=0, dual_regularizer=1, norm_regularizer=1, categorical_action=True)

    running_losses = []
    running_estimates = []

    for step in tqdm(range(num_steps)):
        for env_step, next_env_step in iterator:
            initial_steps_batch = next(init_iterator)
            losses = estimator.train_step(initial_steps_batch, env_step, next_env_step, eval_torch_policy)
            running_losses.append(losses)

        if step % 500 == 0 or step == num_steps - 1:
            estimate = estimator.estimate_average_reward(dataset, eval_torch_policy)
            running_estimates.append(estimate)
            running_losses = []

    dice_estimate = running_estimates[-1]

    # ==== Save results ====
    results = {
        "clipped_is_score": clipped_is_score,
        "weighted_is_score": weighted_is_score,
        "cwpdis_score": cwpdis_score,
        "pdis_score": pdis_score,
        "ris_clipped_is_score": ris_clipped_is_score,
        "ris_weighted_is_score": ris_weighted_is_score,
        "ris_cwpdis_score": ris_cwpdis_score,
        "ris_pdis_score": ris_pdis_score,
        "fqe_score": fqe_score,
        "dice_estimate": dice_estimate
    }

    return results

def run_evaluations(dataset, dataset_name, algo_name):

    model_name_to_results = {}

    # === Load policy ===
    for f in filter(lambda x: f'_{algo_name}_' in x and f'_{dataset_name}_' in x, os.listdir("d3rlpy_logs")):

        if algo_name == 'bc':
            alg = d3rlpy.algos.DiscreteBC.from_json('d3rlpy_logs/%s/params.json' % f)
            alg.load_model('d3rlpy_logs/%s/model_975.pt' % f)
            eval_alg = DiscreteBCEvaluationWrapper(alg)
            eval_torch_alg = DiscreteBCTorchWrapper(alg)
        elif algo_name == 'bcq':
            alg = d3rlpy.algos.DiscreteBCQ.from_json('d3rlpy_logs/%s/params.json' % f)
            alg.load_model('d3rlpy_logs/%s/model_975.pt' % f)
            eval_alg = DiscreteBCQEvaluationWrapper(alg)
            eval_torch_alg = DiscreteBCQTorchWrapper(alg)
        elif algo_name == 'cql':
            alg = d3rlpy.algos.DiscreteCQL.from_json('d3rlpy_logs/%s/params.json' % f)
            alg.load_model('d3rlpy_logs/%s/model_975.pt' % f)
            eval_alg = DiscreteCQLEvaluationWrapper(alg)
            eval_torch_alg = DiscreteCQLTorchWrapper(alg)
        else:
            raise ValueError("Unknown algo name")

        model_hcomb_name = f.split('_2022')[0]
        results = evaluate_policy(alg, eval_alg, eval_torch_alg, dataset)

        model_name_to_results[model_hcomb_name] = results

    # save model_name_to_results in csv
    with open(f'{algo_name}_{dataset_name}_results.csv', 'w') as f:
        f.write('model_name,clipped_is_score,weighted_is_score,cwpdis_score,pdis_score,ris_clipped_is_score,ris_weighted_is_score,ris_cwpdis_score,ris_pdis_score,fqe_score,dice_estimate\n')
        for model_name, results in model_name_to_results.items():
            f.write(f'{model_name},{results["clipped_is_score"]},{results["weighted_is_score"]},{results["cwpdis_score"]},{results["pdis_score"]},{results["ris_clipped_is_score"]},{results["ris_weighted_is_score"]},{results["ris_cwpdis_score"]},{results["ris_pdis_score"]},{results["fqe_score"]},{results["dice_estimate"]}\n')


if __name__ == '__main__':

    # === Load dataset ===
    dataset, sepsis = get_sepsis('mdp-200')
    dataset_name = 'mdp'
    run_evaluations(dataset, dataset_name, 'bc')
    run_evaluations(dataset, dataset_name, 'bcq')
    run_evaluations(dataset, dataset_name, 'cql')

    dataset, sepsis = get_sepsis('pomdp-200')
    dataset_name = 'pomdp'
    run_evaluations(dataset, dataset_name, 'bc')
    run_evaluations(dataset, dataset_name, 'bcq')
    run_evaluations(dataset, dataset_name, 'cql')

