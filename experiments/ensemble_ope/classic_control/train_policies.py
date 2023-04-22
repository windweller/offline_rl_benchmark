import numpy as np
import d3rlpy

def train_policies(dataset_name):
    dataset, _ = d3rlpy.datasets.get_dataset(dataset_name)

if __name__ == '__main__':
    train_policies('cartpole-replay')