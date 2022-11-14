import torch
import torch.nn as nn

class Network(object):
    def layers(self):
        raise NotImplementedError

class ValueNetwork(nn.Module, Network):
    def __init__(self, input_size, hidden_sizes, output_size,
                 activate_final=False):
        super(ValueNetwork, self).__init__()
        self._layers = []
        # Hidden layers
        for size in hidden_sizes:
            self._layers.append(nn.Linear(input_size, size))
            self._layers.append(nn.ReLU())  # why do they choose ReLU...so lazy
            input_size = size
        # Output layer
        self._layers.append(nn.Linear(input_size, output_size))
        if activate_final:
            self._layers.append(nn.ReLU())
        self._layers = nn.Sequential(*self._layers)

    def forward(self, tensor):
        if type(tensor) == tuple:
            tensor = torch.cat(tensor, dim=1)
        return self._layers(tensor)

    def layers(self):
        return self._layers