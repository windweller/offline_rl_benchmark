# Offline RL Benchmark

Preliminary plan:

`envs`:
- Sepsis-POMDP
- Sepsis-MDP
- TutorBot
- HIV
- DKT
- (planned) Robomimic interface

`opes`:
- Various IS estimators
- FQE (D3RLPy)
- BVFT (and various combinations with  FQE)
- (planned) DICE RL
- (planned) SLOPE

`opls`:
- Pessimistic MDP (pMDP)
- BCQ (from D3RLPy)
- MBSQI
- BC (from D3RLPy)
- PG
- BC + PG
- BC + miniPG
- OffPAC
- OPPOSD
- Other D3RLPy algorithms
- Model-based selection (L-MOPO, MML)
- DT

`data_augs`:
- Filter (Filtered-BC)
- Data Augmentation stuff
  - Reward relabeling (Q-DT)
  - Trajectory stitching...

`model_selection`
- K-fold RS
- M-fold CV
- Bootstrap Validation
- Nested CV
- Data-based matching (i.e., reward matching)

`exp`:
- Replicate NeurIPS paper experiments (and show how things are used) (in a fork! In `exp` folder)

Some TODOs:
1. Need to organize both opes/opls into discrete vs. continuous (follow D3RLPy guideline) (not everything will work)

Plan:

(Thurs)
1. ~~Verify PDIS works correctly...~~
2. Add TutorBot...
2. Add pMDP, PG, DiscreteBC, DiscreteMOPO to `algs` folder
3. **Add evaluation wrappers for AWAC/SAC and others**
4. Add model-selection method (this is a hyperparam loop...) (check SKLearn's API design) (works on MDPDataset)
5. Add **data augs**: any-perc BC, and others (define interface first)
6. Add BVFT!! (How does BVFT fit? Probably into `model_selection`)

Other Needs:
1. We need test-cases to make sure OPE is correctly implemented.

Resources/References:
- https://github.com/tinkoff-ai/CORL 