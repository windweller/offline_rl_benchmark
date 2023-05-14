from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
    get_sepsis_subsample_copies, get_sepsis_population_full, get_sepsis_gt, get_sepsis_copies
from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF, load_sepsis_ensemble_mdp_policies, MDP_POLICY_TURE_PERF

from offline_rl.envs.dataset import convert_dataset_for_is_ope
from offline_rl.opes.importance_sampling import compute_pib_pie, importance_sampling_scorer, wis_scorer, \
    cwpdis_scorer, pdis_scorer

from tqdm import tqdm
import pandas as pd
import cvxpy as cp
import numpy as np


dataset, sepsis, data = get_sepsis_gt('mdp-200')
policies = load_sepsis_ensemble_mdp_policies(sepsis)

opt_policy = policies[0]
dataset_with_prob = convert_dataset_for_is_ope(dataset)
score = importance_sampling_scorer(opt_policy, dataset_with_prob.episodes)

print(score)

# train FQE!?