import csv
from tqdm import tqdm

import d3rlpy
from d3rlpy.algos import DiscreteCQL, DiscreteBC, DiscreteBCQ

# hyperparameter grid

newtork_sizes = ['small', 'medium', 'large']

# BCQ (4 hours)
bcq_network_map = {'small': [32], 'medium': [64], 'large': [128]}
bcq_early_stopping = [15, 20, 25]
bcq_special = {'BCQ_threshold': [0.1, 0.3, 0.5]}

bc_network_map = {'small': [32, 32], 'medium': [64, 64], 'large': [128, 128]}
bc_early_stopping = [15, 20, 25]
bc_special = {}

cql_network_map = {'small': [32], 'medium': [64], 'large': [128]}
cql_early_stopping = [15, 20, 25]
cql_special = {'cql_alpha': [0.5, 1.0, 4.0]}

def train_policies(alg_name: str, save_folder:str, batch_size: int = 32, use_gpu: bool = False):
    hyperparam_combs = []
    for ns in newtork_sizes:
        for epoch in eval(alg_name + '_early_stopping'):
            if alg_name == 'bcq':
                for _, threshold in eval(alg_name + '_special').items():
                    hyperparam_combs.append({'network_size': ns,
                                             'threshold': threshold, 'epoch': epoch})
            elif alg_name == 'cql':
                for _, alpha in eval(alg_name + '_special').items():
                    hyperparam_combs.append({'network_size': ns,
                                             'alpha': alpha, 'epoch': epoch})
            else:
                hyperparam_combs.append({'network_size': ns, 'epoch': epoch})

    exp_name_to_hcomb = {}
    result_file = open(f"./{save_folder}/{alg_name}_true_perf.csv", 'w')
    csv_writer = csv.writer(result_file)
    csv_writer.writerow(['Exp_name', 'HComb', 'True Reward Mean'])

    for comb_idx, h_comb in tqdm(enumerate(hyperparam_combs), total=len(hyperparam_combs)):
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
                                   lam=0.75,
                                   action_flexibility=0.05,
                                   use_gpu=use_gpu)