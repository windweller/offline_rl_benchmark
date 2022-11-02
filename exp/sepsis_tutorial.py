from d3rlpy.algos import DQN, DiscreteCQL
from offline_rl.envs.datasets import get_sepsis
from offline_rl.envs.policy_wrappers import QLearningWrapper
from offline_rl.envs.sepsis.env import evaluate_on_sepsis_environment
from sklearn.model_selection import train_test_split

# ===== POMDP setting ======
cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('pomdp-200')
# cql.build_with_env(sepsis)
cql.build_with_dataset(dataset)

train_episodes, test_episodes = train_test_split(dataset, test_size=0.5, random_state=3)
cql.fit(train_episodes, eval_episodes=test_episodes,
            n_epochs=2,
            experiment_name=f"CQL_test_on_sepsis_pomdp_200")

evaluation_policy = QLearningWrapper(cql)
rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)
print(rew)

# ===== MDP setting ======
cql = DiscreteCQL(use_gpu=False)
dataset, sepsis = get_sepsis('mdp-200')
# cql.build_with_env(sepsis)
cql.build_with_dataset(dataset)
evaluation_policy = QLearningWrapper(cql)
rew = evaluate_on_sepsis_environment(sepsis)(evaluation_policy)
print(rew)

# ===== Try Fitting FQE ====
