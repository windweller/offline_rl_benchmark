import numpy as np


class BasicPolicy(object):
    def __init__(self, actions, probs):
        self.actions = actions
        self.probs = np.array(probs)
        self.action_space_dim = len(self.actions)
        assert len(self.actions) == len(self.probs)

    def predict(self, xs, **kw):
        return np.array([self.probs for _ in range(len(xs))])

    def sample(self, xs):
        return self(xs)

    def __call__(self, states):
        return np.random.choice(self.actions, size=len(states), p=self.probs)


