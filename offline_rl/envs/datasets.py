# pylint: disable=unused-import,too-many-return-statements

import os
import random
import re
from typing import List, Tuple
from urllib import request
import shutil

import gym
import numpy as np
import pandas as pd

from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment
import d3rlpy.datasets

from offline_rl.envs.dataset import ProbabilityMDPDataset
from numba import njit, prange

DATA_DIRECTORY = "offline_rl_data"
GITHUB_URL = "https://github.com"

# ======= Sepsis =======

from offline_rl.envs.sepsis.env import Sepsis, load_sepsis_dataset

SEPSIS_URL = f"{GITHUB_URL}/StanfordAI4HI/Split-select-retrain/raw/main/envs/sepsis/sontag_sepsis.zip"

SEPSIS_ENVS = [
    'pomdp-200',
    'pomdp-1000',
    'pomdp-5000',
    'mdp-200',
    'mdp-1000',
    'mdp-5000',
]


def get_sepsis(env_name: str) -> Tuple[ProbabilityMDPDataset, Sepsis]:
    """
    :param env_name: dataset size
    :return:
    """
    assert env_name in SEPSIS_ENVS, print("available env names are: ", SEPSIS_ENVS)

    file_name = 'sontag_sepsis.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading sontag_sepsis.zip into {data_path}...")
        request.urlretrieve(SEPSIS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/sontag_sepsis/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''
    filepath = f"{DATA_DIRECTORY}/sontag_sepsis/marginalized_sepsis_{data_size}_w_noise_05" + appendage + ".csv"
    data = pd.read_csv(filepath)
    dataset = load_sepsis_dataset(data, env)

    return dataset, env


def get_sepsis_weighted(env_name: str, traj_weight: np.array) -> Tuple[ProbabilityMDPDataset, Sepsis]:
    """
    :param env_name: dataset size
    :return:
    """
    assert env_name in SEPSIS_ENVS, print("available env names are: ", SEPSIS_ENVS)

    file_name = 'sontag_sepsis.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading sontag_sepsis.zip into {data_path}...")
        request.urlretrieve(SEPSIS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/sontag_sepsis/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''
    filepath = f"{DATA_DIRECTORY}/sontag_sepsis/marginalized_sepsis_{data_size}_w_noise_05" + appendage + ".csv"
    data = pd.read_csv(filepath)
    dataset = load_sepsis_dataset(data, env, traj_weight=traj_weight)

    return dataset, env


def create_df_copy(num_copies, indices, total_patient_num, patient_subdatasets, traj_name, env):
    datasets = []
    traj_dist = np.zeros(total_patient_num)

    for _ in range(num_copies):

        sampled_patients = np.random.choice(indices, total_patient_num, replace=True)
        sampled_patient_subdatasets = []
        for j, i in enumerate(sampled_patients):
            traj_dist[i] += 1  # adding to the dist
            df = patient_subdatasets[i].copy()
            df[traj_name] = df[traj_name].astype(str) + '_' + str(j)
            sampled_patient_subdatasets.append(df)
        sampled_data = pd.concat(sampled_patient_subdatasets)
        dataset = load_sepsis_dataset(sampled_data, env)

        datasets.append(dataset)

    traj_dist = traj_dist / traj_dist.sum()
    return datasets, traj_dist


def get_sepsis_boostrap_copies(env_name: str, num_copies: int, k_prop: float = 0.2) -> Tuple[
    List[ProbabilityMDPDataset], Sepsis, np.array, float]:
    """
    :param env_name: dataset size
    :param num_copies: number of copies of the dataset
    :return:
        List[ProbabilityMDPDataset]: list of datasets
        Sepsis: environment
        np.array: trajectory distribution (how many times a trajectory is sampled in our procedure)
        float: scale ratio
    """
    assert env_name in SEPSIS_ENVS, print("available env names are: ", SEPSIS_ENVS)

    file_name = 'sontag_sepsis.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading sontag_sepsis.zip into {data_path}...")
        request.urlretrieve(SEPSIS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/sontag_sepsis/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''
    filepath = f"{DATA_DIRECTORY}/sontag_sepsis/marginalized_sepsis_{data_size}_w_noise_05" + appendage + ".csv"
    data = pd.read_csv(filepath)

    traj_name = env.get_trajectory_marking_name()
    patient_subdatasets = []
    for unique_traj in data[traj_name].unique():
        patient_subdatasets.append(data[data[traj_name] == unique_traj])

    # k samples in each bootstrap
    total_patient_num = int(len(patient_subdatasets) * k_prop)
    indices = np.arange(total_patient_num)

    datasets, traj_dist = create_df_copy(num_copies, indices, total_patient_num, patient_subdatasets, traj_name, env)

    return datasets, env, traj_dist, total_patient_num / len(patient_subdatasets)


def get_sepsis_subsample_copies(env_name: str, num_copies: int, percentage: float = 0.5) -> Tuple[
    List[ProbabilityMDPDataset],
    Sepsis, np.array]:
    """
    :param env_name: dataset size
    :param num_copies: number of copies of the dataset
    :return:
    """
    assert env_name in SEPSIS_ENVS, print("available env names are: ", SEPSIS_ENVS)

    file_name = 'sontag_sepsis.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading sontag_sepsis.zip into {data_path}...")
        request.urlretrieve(SEPSIS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/sontag_sepsis/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''
    filepath = f"{DATA_DIRECTORY}/sontag_sepsis/marginalized_sepsis_{data_size}_w_noise_05" + appendage + ".csv"
    data = pd.read_csv(filepath)

    traj_name = env.get_trajectory_marking_name()
    patient_subdatasets = []
    for unique_traj in data[traj_name].unique():
        patient_subdatasets.append(data[data[traj_name] == unique_traj])

    total_patient_num = len(patient_subdatasets)
    indices = np.arange(total_patient_num)

    datasets = []
    traj_dist = np.zeros(total_patient_num)

    for _ in range(num_copies):

        sampled_patients = np.random.choice(indices, int(total_patient_num * percentage), replace=False)
        sampled_patient_subdatasets = []
        for j, i in enumerate(sampled_patients):
            traj_dist[i] += 1
            df = patient_subdatasets[i].copy()
            # df[traj_name] = df[traj_name].astype(str) + '_' + str(j)
            sampled_patient_subdatasets.append(df)
        sampled_data = pd.concat(sampled_patient_subdatasets)
        dataset = load_sepsis_dataset(sampled_data, env)

        datasets.append(dataset)

    traj_dist = traj_dist / traj_dist.sum()

    return datasets, env, traj_dist


# ======= Tutorbot =======

from offline_rl.envs.tutorbot.env import TutorBot, load_tutorbot_dataset

TUTOR_URL = f"{GITHUB_URL}/StanfordAI4HI/Split-select-retrain/raw/main/envs/tutorbot/tutorbot_data.zip"

TUTOR_ENVS = [
    'tutor-200',
    'tutor-80'
]


def get_tutorbot(env_name: str) -> Tuple[ProbabilityMDPDataset, TutorBot]:
    """

    :param env_name: dataset size
    :return:
    """
    assert env_name in TUTOR_ENVS, print("available env names are: ", TUTOR_ENVS)

    file_name = 'tutorbot_data.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading tutorbot_data.zip into {data_path}...")
        request.urlretrieve(TUTOR_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    env = TutorBot(env_name)

    _, data_size = env_name.split('-')
    filepath = f"{DATA_DIRECTORY}/tutorbot_data/student_{data_size}_rand.csv"
    data = pd.read_csv(filepath)
    dataset = load_tutorbot_dataset(data, env)

    return dataset, env


# ===== Sepsis Ensemble ======

SEPSIS_ENS_URL = f"{GITHUB_URL}/StanfordAI4HI/Split-select-retrain/raw/main/envs/sepsis/ens_ope.zip"

SEPSIS_ENS_ENVS = [
    'pomdp-100',
    'pomdp-200',
    'pomdp-500',
    'mdp-100',
    'mdp-200',
    'mdp-500',
]

POLICY_NOISE = ['0', '005', '010', '015', '020']


def get_sepsis_ensemble_full(env_name: str) -> Tuple[ProbabilityMDPDataset, Sepsis]:
    """
    :param env_name: dataset size
    :return:
    """
    assert env_name in SEPSIS_ENS_ENVS, print("available env names are: ", SEPSIS_ENS_ENVS)

    file_name = 'ens_ope.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading ens_ope.zip into {data_path}...")
        request.urlretrieve(SEPSIS_ENS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/ens_ope/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''
    all_df = []
    for noise in POLICY_NOISE:
        filepath = f"{DATA_DIRECTORY}/ens_ope/ens_ope_marginalized_sepsis_{data_size}_w_noise_{noise}" + appendage + ".csv"
        data = pd.read_csv(filepath)
        all_df.append(data)

    data = pd.concat(all_df)
    dataset = load_sepsis_dataset(data, env)

    return dataset, env


def get_sepsis_ensemble_exclude_one(env_name: str) -> Tuple[List[ProbabilityMDPDataset], Sepsis]:
    """
    We give back a list of datasets, a concatenation of all the datasets except one (the sampling dataset)
    :param env_name:
    :return:
    """
    assert env_name in SEPSIS_ENS_ENVS, print("available env names are: ", SEPSIS_ENS_ENVS)

    file_name = 'ens_ope.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading ens_ope.zip into {data_path}...")
        request.urlretrieve(SEPSIS_ENS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/ens_ope/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''

    all_exclude_one_datasets = []
    for exclude_noise in POLICY_NOISE:
        all_df = []
        for noise in POLICY_NOISE:
            if noise == exclude_noise:
                continue
            filepath = f"{DATA_DIRECTORY}/ens_ope/ens_ope_marginalized_sepsis_{data_size}_w_noise_{noise}" + appendage + ".csv"
            data = pd.read_csv(filepath)
            all_df.append(data)

        data = pd.concat(all_df)
        dataset = load_sepsis_dataset(data, env)
        all_exclude_one_datasets.append(dataset)

    return all_exclude_one_datasets, env


def get_sepsis_ensemble_datasets(env_name: str) -> List[pd.DataFrame]:
    assert env_name in SEPSIS_ENS_ENVS, print("available env names are: ", SEPSIS_ENS_ENVS)

    file_name = 'ens_ope.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading ens_ope.zip into {data_path}...")
        request.urlretrieve(SEPSIS_ENS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''

    all_df = []
    for noise in POLICY_NOISE:
        filepath = f"{DATA_DIRECTORY}/ens_ope/ens_ope_marginalized_sepsis_{data_size}_w_noise_{noise}" + appendage + ".csv"
        data = pd.read_csv(filepath)
        all_df.append(data)

    return all_df


def combine_sepsis_ensemble_datasets(env_name: str, datasets: List[pd.DataFrame]) -> Tuple[
    ProbabilityMDPDataset, Sepsis]:
    assert env_name in SEPSIS_ENS_ENVS, print("available env names are: ", SEPSIS_ENS_ENVS)

    file_name = 'ens_ope.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        raise Exception("Must call get_sepsis_ensemble_datasets or other function to download data first")

    if 'pomdp' in env_name:
        assert 'diabetes_idx' not in datasets[0].columns, print("dataset appears to be MDP, but env setup is POMDP")

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/ens_ope/all_states.csv")
    env = Sepsis(env_name, all_states)

    data = pd.concat(datasets)
    dataset = load_sepsis_dataset(data, env)

    return dataset, env


# ===== Sepsis Population ======

SEPSIS_POP_URL = f"{GITHUB_URL}/StanfordAI4HI/Split-select-retrain/raw/main/envs/sepsis/ens_ope_pop.zip"

SEPSIS_POP_ENVS = [
    'pomdp-200',
    'mdp-200',
    'pomdp-1000',
    'mdp-1000'
]


def get_sepsis_population_full(env_name: str, load_first_n: int = 20) -> Tuple[List[ProbabilityMDPDataset],
ProbabilityMDPDataset, Sepsis]:
    """
    :param env_name: dataset size
    :return:
    """
    assert env_name in SEPSIS_POP_ENVS, print("available env names are: ", SEPSIS_POP_ENVS)

    assert load_first_n <= 100, print("load_first_n must be <= 100, we only have 100 datasets in total")

    noise = '005'

    file_name = 'ens_ope_pop.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading ens_ope_pop.zip into {data_path}...")
        request.urlretrieve(SEPSIS_ENS_URL, data_path)
        shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/ens_ope_pop/all_states.csv")
    env = Sepsis(env_name, all_states)

    mdp_type, data_size = env_name.split('-')
    appendage = '_full_states' if mdp_type == 'mdp' else ''

    datasets = []
    dfs = []
    for pop_idx in range(load_first_n):
        filepath = f"{DATA_DIRECTORY}/ens_ope_pop/ens_ope_marginalized_sepsis_{data_size}_w_noise_{noise}_pop_{pop_idx}" + appendage + ".csv"
        data = pd.read_csv(filepath)

        dataset = load_sepsis_dataset(data, env)
        datasets.append(dataset)
        dfs.append(data)

    data = pd.concat(dfs)
    dataset = load_sepsis_dataset(data, env)

    return datasets, dataset, env


# ===== Sepsis Ground Truth ======


def get_sepsis_gt(env_name: str, n=0, seed=None):
    """
    The goal, creating the sampling from the ground truth simulator evaluation
    pass in 'n' with 'pomdp' or 'mdp' as env_name to get a different sample each time
    :param env_name:
    :param n:
    :return:
    """
    assert 'pomdp' in env_name or 'mdp' in env_name, print("acceptable format: 'pomdp', 200 for random sample, "
                                                            "'pomdp-200' for fixed sample ")

    file_name = 'ens_ope_gt.zip'
    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading ens_ope_gt.zip into {data_path}...")
        raise Exception("Download not implemented")
        # request.urlretrieve(SEPSIS_ENS_URL, data_path)
        # shutil.unpack_archive(data_path, DATA_DIRECTORY)

    all_states = pd.read_csv(f"{DATA_DIRECTORY}/ens_ope_gt/all_states.csv")
    env = Sepsis(env_name, all_states)

    # here is a little different
    if '-' in env_name:
        mdp_type, data_size = env_name.split('-')
        data_size = int(data_size)
    else:
        mdp_type = env_name
        data_size = n

    appendage = '_full_states' if mdp_type == 'mdp' else ''

    filepath = f"{DATA_DIRECTORY}/ens_ope_gt/consistent_gt_marginalized_sepsis_5000_c_noise_005_p_005" + appendage + ".csv"
    data = pd.read_csv(filepath)

    traj_name = env.get_trajectory_marking_name()
    patient_subdatasets = []
    for unique_traj in data[traj_name].unique():
        patient_subdatasets.append(data[data[traj_name] == unique_traj])

    total_patient_num = len(patient_subdatasets)
    indices = np.arange(total_patient_num)

    if '-' in env_name:
        # we fix a seed so it's the same result
        if seed is None:
            np.random.seed(1996)
        else:
            np.random.seed(seed)

    if n != 5000:
        sampled_patients = np.random.choice(indices, data_size, replace=False)

        sampled_patient_subdatasets = []
        for j, i in enumerate(sampled_patients):
            df = patient_subdatasets[i].copy()
            # df[traj_name] = df[traj_name].astype(str) + '_' + str(j)
            sampled_patient_subdatasets.append(df)
        sampled_data = pd.concat(sampled_patient_subdatasets)
        dataset = load_sepsis_dataset(sampled_data, env)
        data = sampled_data
    else:
        dataset = load_sepsis_dataset(data, env)

    return dataset, env, data

def get_sepsis_copies(data: pd.DataFrame, env: Sepsis, num_copies: int, k_prop: float = 0.2) -> Tuple[
    List[ProbabilityMDPDataset], Sepsis, np.array, float]:
    """
    :param env_name: dataset size
    :param num_copies: number of copies of the dataset
    :return:
        List[ProbabilityMDPDataset]: list of datasets
        Sepsis: environment
        np.array: trajectory distribution (how many times a trajectory is sampled in our procedure)
        float: scale ratio
    """

    traj_name = env.get_trajectory_marking_name()
    patient_subdatasets = []
    for unique_traj in data[traj_name].unique():
        patient_subdatasets.append(data[data[traj_name] == unique_traj])

    # k samples in each bootstrap
    total_patient_num = int(len(patient_subdatasets) * k_prop)
    indices = np.arange(total_patient_num)

    datasets, traj_dist = create_df_copy(num_copies, indices, total_patient_num, patient_subdatasets, traj_name, env)

    return datasets, env, traj_dist, total_patient_num / len(patient_subdatasets)