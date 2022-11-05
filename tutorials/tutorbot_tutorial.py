"""
Technically this should be PG methods
but currently we use CQL isntead
"""

from d3rlpy.algos import DQN, DiscreteCQL
from d3rlpy.ope import DiscreteFQE
from offline_rl.envs.datasets import get_tutorbot
from offline_rl.algs.discrete_policy_evaluation_wrappers import QLearningEvaluationWrapper
from offline_rl.envs.tutorbot.env import evaluate_on_tutorbot_environment
from sklearn.model_selection import train_test_split

from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer

# ===== Policy Learning ======
cql = DiscreteCQL(use_gpu=False)
dataset, tutor = get_tutorbot('tutor-200')
# cql.build_with_env(sepsis)
cql.build_with_dataset(dataset)

train_episodes, test_episodes = train_test_split(dataset, test_size=0.5, random_state=3)
cql.fit(train_episodes, eval_episodes=test_episodes,
            n_epochs=2,
            experiment_name=f"CQL_test_on_tutorbot_pomdp_200")

rew = evaluate_on_tutorbot_environment(tutor)(cql)
print(rew)
# 1.4812252109180077 (this is higher than the paper reported result)

# ===== Try Fitting FQE ====

fqe = DiscreteFQE(algo=cql)
fqe.build_with_dataset(dataset)

metrics = fqe.fit(test_episodes, n_epochs=2,
        scorers={
            'init_value': initial_state_value_estimation_scorer
        })
print(metrics)