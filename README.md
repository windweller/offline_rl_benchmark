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
- Other D3RLPy algorithms
- Model-based selection (L-MOPO, MML)

`model_selection`
- K-fold RS
- M-fold CV
- Bootstrap Validation
- Nested CV
- Data-based matching (i.e., reward matching)

`exp`:
- Replicate NeurIPS paper experiments (and show how things are used)

Some TODOs:
1. Need to organize both opes/opls into discrete vs. continuous (follow D3RLPy guideline) (not everything will work)

Plan:

(Monday, Tuesday)
1. ~~Add Sepsis-POMDP, MDP to D3RLPy format.~~
2. Test if FQE works out of box with Sepsis... 
3. Add WIS style OPE
4. Add DiscreteBCQ, pMDP to `algs` folder
5. See if we can train FQE on this interface...