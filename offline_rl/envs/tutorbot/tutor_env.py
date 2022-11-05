import pandas as pd
from gym import spaces
import numpy as np

from collections import deque

import torch

import random
import gym
from gym import spaces
from gym.utils import seeding


class StudentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, prefer_high_or_low_pre=None):
        self.action_space = spaces.Discrete(2)
        self.prefer_high_or_low_pre = prefer_high_or_low_pre
        self.sample_initial_state()

    def sample_initial_state(self):
        # natural distribution
        if self.prefer_high_or_low_pre is None:
            high_or_low_pre = 'high' if np.random.rand() < 0.35 else 'low'
        else:
            high_or_low_pre = self.prefer_high_or_low_pre

        # 35% to go into high
        if high_or_low_pre == 'high':
            dist = [(5.0, 0.1),
                    (6.0, 0.21666666666666667),
                    (7.0, 0.21666666666666667),
                    (8.0, 0.4666666666666667)]
            # [5,6,7,8]
            self.pre_test = np.random.choice([d[0] for d in dist], 1, p=[d[1] for d in dist])
        else:
            dist = [(0.0, 0.036036036036036036),
                    (1.0, 0.08108108108108109),
                    (2.0, 0.2072072072072072),
                    (3.0, 0.34234234234234234),
                    (4.0, 0.3333333333333333)]
            # self.pre_test = np.random.randint(0, 4+1) #[0,1,2,3, 4]
            self.pre_test = np.random.choice([d[0] for d in dist], 1, p=[d[1] for d in dist])

        self.pre_test = float(self.pre_test)

        # RL brought improvement:
        # for low group: if 1 ~ p(improv), pull rl_improv, otherwise baseline_improv
        self.baseline_improv = {
            0.0: 0.67,
            1.0: 0.86,
            2.0: 1.24,  # this number is fake, mean of 1.0 and 3.0
            3.0: 1.67,
            4.0: 1.65,  # fake, swapped baseline and
            # 5.0: 1.0
        }
        self.rl_improv = {
            0.0: 3.0,
            1.0: 2.11,
            2.0: 2.13,
            3.0: 2.39,
            4.0: 1.89,
            # 5.0: 2.17,
        }
        # for high group: higher rew - annoyance penalty
        self.high_perf_reward = {
            5.0: 2.17,
            6.0: 1.75,
            7.0: 0.91,
            8.0: 0
        }

        # created by subtracting between RL condition and baseline condition
        self.annoyance_penalty = 0.15  # 0.2 * 3 (all 3 actions are guided prompt, it will drop)

        # dynamics: [0,1,2,3,4,5] one dynamic, [6,7,8] another dynamic

        # sample lengths
        self.T = int(np.round(7 - 0.46 * self.pre_test + np.random.randint(-1, 2)))
        self.curr_step = 0
        # self.failed_attempts = np.random.choice([0, 1, 2])  # we start with some...
        self.chance_to_improve = 0
        self.past_actions = deque(maxlen=4)  # 4-th order

        # 0.6, means if the last 3 actions are consecutively "guided prompt", chance of learning
        self.anxiety_coeff = [0, -0.05, -0.2, -0.5]
        self.thinking_coeff = [0.5, 0.3, 0.2, 0]

        # we actually need to send out a warning sign, saying this is the last step, otherwise agent can't learn the behavior

        # data heterogenity:
        # 1. Based on pre-test, the end rewards are different
        # 2. Based on pre-test, learning dynamics are different too
        self.last_step = False
        if self.pre_test >= 5: # >= 5
            self.final_reward = self.high_perf_reward[self.pre_test]
        else:
            self.final_reward = None

        # for high-perf, anxiety and thinking don't decide, so we'll sample a random anxiety
        if self.pre_test >= 5:
            self.anxiety = 0.2
            self.thinking = 0.5
        else:
            self.anxiety = 0.3
            self.thinking = 0.3

    def __repr__(self):
        return f"StudentEnv pre_test={self.pre_test} T={self.T} anxiety={self.anxiety} thinking={self.thinking}"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: float):

        self.past_actions.append(action)

        done = None
        # action = 0: encourage, action = 1: guided prompt; action = 2: hint.
        if self.pre_test >= 5: # >= 5
            # student gets annoyed
            self.final_reward -= self.annoyance_penalty * action
            self.anxiety = max(0, self.anxiety + np.random.choice([-0.1, 0, 0.1]))
            self.thinking = max(0, self.anxiety + np.random.choice([-0.1, 0, 0.1]))
        else:
            # different dynamic
            # i = 0 is the oldest item, furtherest away
            if len(self.past_actions) >= 4:
                self.anxiety, self.thinking = 0, 0
                for i, past_action in enumerate(self.past_actions):
                    self.anxiety += self.anxiety_coeff[i] * past_action
                    self.thinking += self.thinking_coeff[i] * past_action
            else:
                for i, past_action in enumerate(self.past_actions):
                    self.anxiety += self.anxiety_coeff[3 - i] * past_action
                    self.thinking += self.thinking_coeff[3 - i] * past_action

        self.anxiety = max(-1, min(0, self.anxiety))
        self.thinking = max(0, min(1, self.thinking))

        # after getting the states, count as final
        self.curr_step += 1

        last_step = True if self.curr_step == self.T - 1 else False

        if done is None:
            done = self.T == self.curr_step

        if done:
            if self.pre_test >= 5:
                reward = self.final_reward
                if self.pre_test == 5:
                    reward = max(0, np.random.normal(self.final_reward, 0.5)) # 0.98
                elif self.pre_test == 6:
                    reward = max(0, np.random.normal(self.final_reward, 0.5)) # 0.5
                elif self.pre_test == 7:
                    reward = max(0, np.random.normal(self.final_reward, 0.44)) # 0.44
                elif self.pre_test == 8:
                    reward = 0

            else:
                chance_to_improve = self.anxiety + self.thinking
                if np.random.rand() <= chance_to_improve:
                    mean = self.rl_improv[self.pre_test]
                    reward = max(0, np.random.normal(mean, 1))
                else:
                    mean = self.baseline_improv[self.pre_test]
                    reward = max(0, np.random.normal(mean, 0.4))
        else:
            reward = 0

        state = [self.pre_test, self.anxiety, self.thinking, int(last_step)]

        return np.array(state), reward, done, {}

    def reset(self):
        self.sample_initial_state()
        state = [self.pre_test, self.anxiety, self.thinking, 0]
        return np.array(state)


# needs to use a wrapper to go along w/ our MDP
class MLPPolicyWrapper(object):
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, state, sample=True):
        p = self.policy.get_action_probability(state)
        if sample:
            action = np.random.choice([0., 1., 2.], p=p)
        else:
            action = np.argmax(p)
        return action

    def get_action_prob(self, state):
        state = torch.from_numpy(state).float()
        p = self.policy.get_action_probability(state.view(1, -1))
        return p.squeeze().numpy()

    def sample_action(self, p):
        action = np.random.choice([0., 1., 2.], p=p)
        return action

# our dataset is not that big
# a fully random policy might be enough...
class RandomPolicy(object):
    def get_action(self, state, sample=True):
        return np.random.choice([0., 1., 2.])

    def get_action_probability(self, state):
        return torch.from_numpy(np.array([1/3, 1/3, 1/3]))

class RandomPolicyWithNoise(object):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale

    def get_action(self, state, sample=True):
        return np.random.choice([0., 1., 2.])

    def get_action_probability(self, state):
        prob = [1/3 + random.random() * self.noise_scale, 1/3 + random.random() * self.noise_scale,
                1/3 + random.random() * self.noise_scale]
        prob = np.array(prob)
        prob = prob / prob.sum()
        return torch.from_numpy(prob)


class NearOptimalPolicy_NoNoise(object):
    def __init__(self, noise=0):
        self.noise = noise

    def get_action_probability(self, state):
        state = state.squeeze()
        if state[0] >= 5:
            return torch.from_numpy(np.array([1 - self.noise, 0 + self.noise / 2, 0 + self.noise / 2]))
        else:
            if state[-1] != 1:
                return torch.from_numpy(np.array([0 + self.noise / 2, 0 + self.noise / 2, 1 - self.noise]))
            else:
                return torch.from_numpy(np.array([1 - self.noise, 0 + self.noise / 2, 0 + self.noise/2]))


class NearOptimalPolicy(object):
    def __init__(self, sub_opt=0.):
        self.sub_opt = sub_opt

    def get_action_probability(self, state):
        state = state.squeeze()
        rand_noise = random.random() * 0.1  # at most 10% noise; if set sub_opt = 0.6, then there's a chance bad action is at top...
        if state[0] >= 5:
            return torch.from_numpy(np.array([1 - self.sub_opt - rand_noise, 0 + self.sub_opt / 2 + rand_noise / 2,
                                              0 + self.sub_opt / 2 + rand_noise / 2]))
        else:
            if state[-1] != 1:
                return torch.from_numpy(np.array([0 + self.sub_opt / 2 + rand_noise / 2, 0 + self.sub_opt / 2 + rand_noise / 2, 1 - self.sub_opt - rand_noise]))
            else:
                return torch.from_numpy(np.array([1 - self.sub_opt - rand_noise, 0 + self.sub_opt / 2 + rand_noise / 2,
                                              0 + self.sub_opt / 2 + rand_noise / 2]))

def policy_eval_on_true_mdp(policy):
    np.random.seed(1234)
    random.seed(1234)

    policy_wrapper = MLPPolicyWrapper(policy)
    stu = StudentEnv()

    total_reward = 0

    for _ in range(1000):
        done = False
        state = stu.reset()

        while not done:
            p = policy_wrapper.get_action_prob(state)
            action = policy_wrapper.sample_action(p)
            next_state, r, done, info = stu.step(action=action)

            # traj.append([state, action, next_state, r, p])
            state = next_state

        total_reward += r

    return total_reward / 1000

def sample_trajectories(policy, N):
    # np.random.seed(1234)
    # random.seed(1234)

    policy_wrapper = MLPPolicyWrapper(policy)
    stu = StudentEnv()

    data = []
    for stu_id in range(N):
        done = False
        state = stu.reset()

        while not done:
            p = policy_wrapper.get_action_prob(state)
            action = policy_wrapper.sample_action(p)

            next_state, r, done, info = stu.step(action=action)

            # traj.append([state, action, next_state, r, p])
            data.append([stu_id, state[0], state[1], state[2], state[3], action, r, p[0], p[1], p[2]])
            state = next_state

    # convert to pandas DataFrame
    data_dict = {}
    data = np.array(data)
    for i, name in enumerate(['user_id', 'pre', 'anxiety', 'thinking', 'last_step', 'action',
       'reward', 'p_encourage', 'p_hint', 'p_guided_prompt']):
        data_dict[name] = data[:, i]
    df = pd.DataFrame(data=data_dict)

    return df

if __name__ == '__main__':
    # we can use this to sample from student Env
    save_dir = "./student_simulator_data/"
    policy = RandomPolicy()

    total_df = sample_trajectories(RandomPolicy(), N=200)
    total_df.to_csv(save_dir + "student_200_rand.csv", index=False)

    policy_eval_on_true_mdp(NearOptimalPolicy(0.5))

    random.seed(4321)
    total_df = sample_trajectories(NearOptimalPolicy(0.5), N=200)