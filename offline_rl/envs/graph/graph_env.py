import numpy as np
import scipy.signal as signal
import gym
from gym import spaces
import numpy as np
import d3rlpy

from offline_rl.algs.policy_evaluation_wrappers import *
from d3rlpy.algos.base import *
from gym.envs.registration import register
from d3rlpy.metrics.scorer import evaluate_on_environment

class GraphEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                     make_pomdp = False,
                     number_of_pomdp_states = 2,
                     transitions_deterministic=True,
                     max_length = 2,
                     sparse_rewards = False,
                     stochastic_rewards = False):
        super().__init__()
        self.observation_space = spaces.Box(low=np.array([0]),
                                   high=np.array([2 * max_length]),
                                   dtype=np.int)
        self.action_space = spaces.Box(low=np.array([0]),
                                   high=np.array([1]),
                                   dtype=np.int)

        self.allowable_actions = [0,1]
        self.n_actions = len(self.allowable_actions)
        self.n_dim = 2*max_length



        self.make_pomdp = make_pomdp
        self.number_of_pomdp_states = number_of_pomdp_states

        split = np.array_split(np.arange(2, 2*max_length)-1, number_of_pomdp_states-1)

        self.state_to_pomdp_state = {}
        for pomdp_state,states in enumerate(split):
            for state in states:
                self.state_to_pomdp_state[state] = pomdp_state

        self.state_to_pomdp_state[0] = 0
        self.state_to_pomdp_state[2*max_length-1] = number_of_pomdp_states-1

        self.transitions_deterministic = transitions_deterministic
        self.slippage = .25
        self.max_length = max_length
        self.sparse_rewards = sparse_rewards
        self.stochastic_rewards = stochastic_rewards
        self.reward_overwrite = None # only for simulator work
        self.absorbing_state = None # only for simulator work
        self.reset()

    def overwrite_rewards(self, new_r):
        self.reward_overwrite = new_r

    def set_absorb(self, absorb):
        self.absorbing_state = absorb

    def num_states(self):
        return self.n_dim

    def pos_to_image(self, x):
        '''latent state -> representation '''
        return x

    def reset(self):
        self.state = 0
        self.done = False
        return np.array([self.state])

    def step(self, action):
        assert action in self.allowable_actions
        assert not self.done, 'Episode Over'
        reward = 0 if not self.stochastic_rewards else np.random.randn()
        prev_state = self.state_to_pomdp_state[self.state] if self.make_pomdp else self.state

        if self.state == (2*self.max_length-3):
            reward = 1 if not self.stochastic_rewards else np.random.randn()+1
            # reward = 0
            self.state = 0 #2*self.max_length-1
            self.done = True
        elif self.state == (2*self.max_length-2):
            reward = -1 if not self.stochastic_rewards else np.random.randn()-1
            # reward = 0
            self.state = 0 #2*self.max_length-1
            self.done = True
        else:
            if self.state == 0:
                if action == 0:
                    if self.transitions_deterministic:
                        self.state = self.state + 1
                    else:
                        self.state = int(np.random.choice([self.state+1,self.state+2], p = [1-self.slippage,self.slippage]))
                else:
                    if self.transitions_deterministic:
                        self.state = self.state + 2
                    else:
                        self.state = int(np.random.choice([self.state+2,self.state+1], p = [1-self.slippage,self.slippage]))
            else:
                if action == 0:
                    if self.transitions_deterministic:
                        if self.state % 2 == 1:
                            self.state = self.state + 2
                        else:
                            self.state = self.state + 1
                    else:
                        if self.state % 2 == 1:
                            self.state = int(np.random.choice([self.state+2,self.state+3], p = [1-self.slippage,self.slippage]))
                        else:
                            self.state = int(np.random.choice([self.state+1,self.state+2], p = [1-self.slippage,self.slippage]))
                else:
                    if self.transitions_deterministic:
                        if self.state % 2 == 1:
                            self.state = self.state + 3
                        else:
                            self.state = self.state + 2
                    else:
                        if self.state % 2 == 1:
                            self.state = int(np.random.choice([self.state+3,self.state+2], p = [1-self.slippage,self.slippage]))
                        else:
                            self.state = int(np.random.choice([self.state+2,self.state+1], p = [1-self.slippage,self.slippage]))

            if not self.sparse_rewards:


                if self.state % 2 == 1:
                    rew = 1 if not self.stochastic_rewards else np.random.randn()+1
                    reward = rew
                else:
                    rew = -1 if not self.stochastic_rewards else np.random.randn()-1
                    reward = rew
            # else:
            #     if self.state == 2*self.max_length-3:
            #         reward = 1
            #     elif self.state == 2*self.max_length-2:
            #         reward = -1
            #     else:
            #         reward = 0

                # reward = 1 if self.state == 2*self.max_length-1

        state = self.state_to_pomdp_state[self.state] if self.make_pomdp else self.state

        if self.reward_overwrite is not None:
            key = tuple([int(prev_state), int(action), int(state)]) if not self.done else tuple([prev_state, action, self.absorbing_state])
            # key = tuple([int(prev_state), int(action)]) if not self.done else tuple([prev_state, action])
            if key in self.reward_overwrite:
                try:
                    reward = np.random.choice(list(self.reward_overwrite[key]), p=list(self.reward_overwrite[key].values()))
                except:
                    import pdb; pdb.set_trace()
            else:
                reward = 0

        if self.make_pomdp:
            # only reveal state, not internal state (POMDP)
            return np.array([state]), reward, self.done, {}
        else:
            return np.array([self.state]), reward, self.done, {}

    def render(self, a=None, r=None, return_arr=False):
        start_state = 1 if self.state == 0 else 0
        state = np.zeros(2*self.max_length-2)
        end_state = 1 if self.state == (2*self.max_length-1) else 0

        if not start_state and not end_state:
            state[self.state-1] = 1

        if return_arr:
            return start_state, state.reshape(2,self.max_length-1, order='F'), end_state
        else:

            print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[0].astype(int).astype(str).tolist()), '  ')
            if (a is not None) and (r is not None):
                print(start_state, ' '*((2*(self.max_length-2))+1), end_state, ' (a,r): ', (a,r), '.  If POMDP, End state: ', end_state)
            else:
                print(start_state, ' '*((2*(self.max_length-2))+1), end_state)
            print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[1].astype(int).astype(str).tolist()), '  ')
            print('\n')
            # print([start_state], [end_state], state.reshape(2,self.max_length-1, order='F'), )

    def calculate_exact_value_of_policy(self, pi_e, gamma):
        # Exact
        # rewards = []
        # if (self.transitions_deterministic):
        #     rew = [(+1)*(pi_e.probs[0]) + (-1)*(pi_e.probs[1])]
        #     if not self.sparse_rewards:
        #         rewards.append(rew*self.max_length)
        #     else:
        #         rewards.append([0]*(self.max_length-1) + rew)

        # else:
        #     rew = [(+1)*(pi_e.probs[0]*(1-self.slippage) + pi_e.probs[1]*(self.slippage)) + (-1)*(pi_e.probs[0]*(self.slippage) + pi_e.probs[1]*(1-self.slippage))]
        #     if not self.sparse_rewards:
        #         rewards.append(rew*self.max_length)
        #     else:
        #         rewards.append([0]*(self.max_length-1) + rew)

        # Approx
        evaluation = []
        for i in range(5000):
            done = False
            state = self.reset()
            # env.render()
            rewards = []
            while not done:
                action = pi_e([state])
                # print(action)
                next_state, reward, done = self.step(action)
                # env.render()
                state = next_state
                rewards.append(reward)

            evaluation.append(rewards)

        true = np.mean([self.discounted_sum(rew, gamma) for rew in np.array(evaluation)])

        return true

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]

    def close(self):
        pass

class GraphPolicy(AlgoBase, DiscreteProbabilisticPolicyProtocol):

    def __init__(self, noise_level = 0, noise_fn = lambda x: np.zeros_like(x), env = None):
        super().__init__(
            batch_size=1,
            n_frames=4,
            n_steps=1,
            gamma=0.0,
            scaler=None,
            action_scaler=None,
            kwargs={},
        )
        self.noise_level = noise_level
        self.noise_fn = noise_fn
        if env:
            n_s = len(env.state_to_pomdp_state) if not env.make_pomdp else env.number_of_pomdp_states
            self.policy_map = dict(zip(range(n_s), self.predict_action_probabilities(np.arange(n_s))))

    def predict(self, x):
        x = np.array(x)
        # print(x, (np.random.uniform(size = x.shape) > 1 - self.noise_level - self.noise_fn(x)).astype(int))
        return (np.random.uniform(size = x.shape) > 1 - self.noise_level - self.noise_fn(x)).astype(int)

    def predict_action_probabilities(self, x):
        x = np.array(x)
        return np.vstack([1 - self.noise_level - self.noise_fn(x), self.noise_level + self.noise_fn(x)]).T

    def collect(self, env, buf, n_steps = 1e3):
        n_collect = 0
        while n_collect < n_steps:
            observation, done = env.reset(), False
            while not done and n_collect < n_steps:
                action = self.predict(observation)
                next_observation, reward, done, _ = env.step(action) 
                buf.append(observation, action, reward, done)
                observation = next_observation
                n_collect += 1

if 'GraphEnv-v0' not in gym.envs.registry.env_specs:
    # del gym.envs.registry.env_specs['GraphEnv-v0']
    register(
        id='GraphEnv-v0',
        entry_point=GraphEnv,
    )

### Example usage:

if __name__ == '__main__':
    env = gym.make('GraphEnv-v0', max_length = 100)

    policy1 = GraphPolicy(env = env)
    policy2 = GraphPolicy(noise_level = 0.25, env = env)
    policy3 = GraphPolicy(noise_level = 0.75, env = env)
    policy4 = GraphPolicy(noise_fn = lambda s: np.where(s % 2 == 0, 0, 0.75), env = env)
    policy5 = GraphPolicy(noise_fn = lambda s: np.where(s % 2 == 0, 0.2, 0.1), env = env)

    eval_fn = evaluate_on_environment(env, gamma = 1)
    print("Policy 1 (optimal) return:", eval_fn(policy1))
    print("Policy 2 (noise: 0.25) return:", eval_fn(policy2))
    print("Policy 3 (noise: 0.75) return:", eval_fn(policy3))
    print("Policy 4 (biased noise: 0.0, 0.75) return:", eval_fn(policy4))
    print("Policy 5 (biased noise: 0.2, 0.1) return:", eval_fn(policy5))

    buf = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000, env=env)
    policy1.collect(env, buf, n_steps=1000)
    dataset = buf.to_mdp_dataset()
    print("Collected", len(dataset), "episodes from policy 1!")
