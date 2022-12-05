import torch
import torch.nn as nn
import torch.functional as F

from typing import Optional, Sequence

class Network(object):
    def layers(self):
        raise NotImplementedError

class Square(nn.Module):
    r"""
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.square(input)

class ValueNetwork(nn.Module, Network):
    def __init__(self, input_size, hidden_sizes, output_size,
                 zeta_pos=False):
        super(ValueNetwork, self).__init__()
        self._layers = []
        # Hidden layers
        for size in hidden_sizes:
            self._layers.append(nn.Linear(input_size, size))
            self._layers.append(nn.ReLU())  # why do they choose ReLU...so lazy
            input_size = size
        # Output layer
        self._layers.append(nn.Linear(input_size, output_size))
        if zeta_pos:
            self._layers.append(Square())
        self._layers = nn.Sequential(*self._layers)

    def forward(self, tensor):
        if type(tensor) == tuple:
            tensor = torch.cat(tensor, dim=1)
        return self._layers(tensor)

    def layers(self):
        return self._layers

class ValueNetworkWithAction(nn.Module, Network):
    def __init__(self, input_size, hidden_sizes, output_size,
                 zeta_pos=False):
        super(ValueNetworkWithAction, self).__init__()
        self._layers = []
        # Hidden layers
        for size in hidden_sizes:
            self._layers.append(nn.Linear(input_size, size))
            self._layers.append(nn.GELU())  # why do they choose ReLU...so lazy
            input_size = size
        # Output layer
        self._layers.append(nn.Linear(input_size, output_size))
        if zeta_pos:
            self._layers.append(Square())
        self._layers = nn.Sequential(*self._layers)

    def forward(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        tensor = torch.cat([state, action], dim=1)
        return self._layers(tensor)

    def layers(self):
        return self._layers
