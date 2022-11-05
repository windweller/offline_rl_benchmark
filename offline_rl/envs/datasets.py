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

from offline_rl.envs.dataset import DiscreteProbabilityMDPDataset

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

def get_sepsis(env_name: str) -> Tuple[DiscreteProbabilityMDPDataset, Sepsis]:
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

# ======= Tutorbot =======

from offline_rl.envs.tutorbot.env import TutorBot, load_tutorbot_dataset

TUTOR_URL = f"{GITHUB_URL}/StanfordAI4HI/Split-select-retrain/raw/main/envs/tutorbot/tutorbot_data.zip"

TUTOR_ENVS = [
    'tutor-200',
    'tutor-80'
]

def get_tutorbot(env_name: str) -> Tuple[DiscreteProbabilityMDPDataset, TutorBot]:
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