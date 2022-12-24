import os
import csv
import json
from tqdm import tqdm

import d3rlpy
from d3rlpy.algos import DiscreteCQL, DiscreteBC, DiscreteBCQ

from offline_rl.envs.datasets import get_sepsis
from offline_rl.algs.policy_evaluation_wrappers import DiscreteCQLEvaluationWrapper, DiscreteBCEvaluationWrapper, \
                                                       DiscreteBCQEvaluationWrapper
from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment


from sklearn.model_selection import train_test_split

# hyperparameter grid

newtork_sizes = ['small', 'medium', 'large']

# BCQ (4 hours)
bcq_network_map = {'small': [32], 'medium': [64], 'large': [128]}
bcq_early_stopping = [15, 25]
bcq_special = {'BCQ_threshold': [0.1, 0.5]}

bc_network_map = {'small': [32, 32], 'medium': [64, 64], 'large': [128, 128]}
bc_early_stopping = [15, 25]
bc_special = {}

cql_network_map = {'small': [32], 'medium': [64], 'large': [128]}
cql_early_stopping = [15, 25]
cql_special = {'cql_alpha': [0.5, 1.0, 4.0]}

def train_policies(dataset, sepsis, alg_name: str, save_folder:str, dataset_name: str, batch_size: int = 32, use_gpu: bool = False):
    hyperparam_combs = []
    for ns in newtork_sizes:
        for epoch in eval(alg_name + '_early_stopping'):
            if alg_name == 'bcq':
                for _, thresholds in eval(alg_name + '_special').items():
                    for threshold in thresholds:
                        hyperparam_combs.append({'network_size': ns,
                                                'threshold': threshold, 'epoch': epoch})
            elif alg_name == 'cql':
                for _, alphas in eval(alg_name + '_special').items():
                    for alpha in alphas:
                        hyperparam_combs.append({'network_size': ns,
                                             'alpha': alpha, 'epoch': epoch})
            else:
                hyperparam_combs.append({'network_size': ns, 'epoch': epoch})

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.5, random_state=3)

    # create folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    exp_name_to_hcomb = {}
    result_file = open(f"./{save_folder}/{alg_name}_{dataset_name}_true_perf.csv", 'w')
    csv_writer = csv.writer(result_file)
    csv_writer.writerow(['Exp_name', 'HComb', 'True Reward Mean'])

    for comb_idx, h_comb in tqdm(enumerate(hyperparam_combs), total=len(hyperparam_combs)):
        exp_name = f"{dataset_name}_{alg_name}_hcomb_{comb_idx}"
        exp_name_to_hcomb[exp_name] = h_comb
        if alg_name == 'bcq':
            vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory(h_comb['network_size'])
            rl_encoder = d3rlpy.models.encoders.VectorEncoderFactory(h_comb['network_size'])

            bcq = d3rlpy.algos.DiscreteBCQ(actor_encoder_factory=rl_encoder,
                                   actor_learning_rate=1e-4,
                                   critic_encoder_factory=rl_encoder,
                                   critic_learning_rate=1e-4,
                                   imitator_encoder_factory=vae_encoder,
                                   imitator_learning_rate=1e-4,
                                   batch_size=batch_size,
                                   beta=h_comb['threshold'],
                                   action_flexibility=0.05,
                                   use_gpu=use_gpu)
            bcq.build_with_dataset(dataset)

            bcq.fit(train_episodes,
                    eval_episodes=test_episodes,
                    n_epochs=h_comb['epoch'],
                    experiment_name=exp_name)


            hcomb = f"{h_comb['network_size']}_{h_comb['threshold']}_{h_comb['epoch']}"
            # evaluation_policy = QLearningEvaluationWrapper(cql)
            evaluation_policy = DiscreteBCQEvaluationWrapper(bcq)
            rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)
        elif alg_name == 'cql':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory(h_comb['network_size'])
            cql = d3rlpy.algos.DiscreteCQL(actor_learning_rate=1e-4,
                                   critic_learning_rate=3e-4,
                                   temp_learning_rate=1e-4,
                                   actor_encoder_factory=encoder,
                                   critic_encoder_factory=encoder,
                                   batch_size=batch_size,
                                   alpha=h_comb['alpha'],
                                   use_gpu=use_gpu)
            cql.build_with_dataset(dataset)

            cql.fit(train_episodes,
                    eval_episodes=test_episodes,
                    n_epochs=h_comb['epoch'],
                    experiment_name=exp_name)

            hcomb = f"{h_comb['network_size']}_{h_comb['alpha']}_{h_comb['epoch']}"
            evaluation_policy = DiscreteCQLEvaluationWrapper(cql)
            rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)
        elif alg_name == 'bc':
            encoder = d3rlpy.models.encoders.VectorEncoderFactory(h_comb['network_size'])
            bc = d3rlpy.algos.DiscreteBC(learning_rate= 1e-3,
                                         batch_size=batch_size,
                                         use_gpu=use_gpu)
            bc.build_with_dataset(dataset)

            bc.fit(train_episodes,
                   eval_episodes=test_episodes,
                   n_epochs=h_comb['epoch'],
                   experiment_name=exp_name)
            # logdir=f"{save_folder}"

            hcomb = f"{h_comb['network_size']}_{h_comb['epoch']}"
            evaluation_policy = DiscreteBCEvaluationWrapper(bc)
            rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)

        else:
            raise NotImplementedError

        csv_writer.writerow([exp_name, hcomb, rew])
        result_file.flush()

    result_file.close()
    json.dump(exp_name_to_hcomb, open(f"./{save_folder}/{alg_name}_hcombs.json", 'w'))

if __name__ == '__main__':
    pass
    dataset, sepsis = get_sepsis('pomdp-200')
    train_policies(dataset, sepsis, 'cql', 'cql_results', 'sepsis_pomdp_200', batch_size=32, use_gpu=False)
    train_policies(dataset, sepsis, 'bcq', 'bcq_results', 'sepsis_pomdp_200', batch_size=32, use_gpu=False)
    train_policies(dataset, sepsis, 'bc', 'bc_results', 'sepsis_pomdp_200', batch_size=32, use_gpu=False)

    dataset, sepsis = get_sepsis('mdp-200')
    train_policies(dataset, sepsis, 'cql', 'cql_results', 'sepsis_mdp_200', batch_size=32, use_gpu=False)
    train_policies(dataset, sepsis, 'bcq', 'bcq_results', 'sepsis_mdp_200', batch_size=32, use_gpu=False)
    train_policies(dataset, sepsis, 'bc', 'bc_results', 'sepsis_mdp_200', batch_size=32, use_gpu=False)