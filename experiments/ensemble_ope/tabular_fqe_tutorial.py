from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from offline_rl.opes.tabular_fqe import TabularFQE

if __name__ == '__main__':
    # load in a sepsis policy and evaluate here
    from offline_rl.envs.sepsis.behavior_policy import load_sepsis_ensemble_policies, POLICY_TRUE_MEAN_PERF, \
        load_sepsis_ensemble_mdp_policies, MDP_POLICY_TURE_PERF
    from offline_rl.envs.datasets import get_sepsis, get_sepsis_boostrap_copies, get_sepsis_ensemble_datasets, \
        get_sepsis_subsample_copies, get_sepsis_population_full, get_sepsis_gt, get_sepsis_copies

    dataset, sepsis, data = get_sepsis_gt('mdp', 2000)
    policies = load_sepsis_ensemble_mdp_policies(sepsis)

    # 0.2593469804660408 for 5th policy
    # 0.2912562257101395 for 4th policy
    # 0.35768663627414093 for 1st policy

    # pomdp
    # 5th policy: -0.2214229484081978  -0.10
    # 1st policy: 0.4259569033009309  -0.05  # 0.37 #
    opt_policy = policies[0]

    fqe = TabularFQE(algo=opt_policy)
    fqe.build_with_env(sepsis)

    # print(opt_policy.policy_map)
    # print((1, 1, 1, 1, 1, 1, 2, 1) in opt_policy.policy_map)
    # print((1, 1, 1, 1, 1, 1, 2, 1, 0) in fqe.Q)

    fqe.fit(dataset.episodes, n_epochs=10)
    #
    # # print(sum(fqe.Q.values()))
    #
    score = initial_state_value_estimation_scorer(fqe, dataset.episodes)
    print(score)
