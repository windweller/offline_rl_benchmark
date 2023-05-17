#!/bin/bash
#
#SBATCH --job-name=anie
#
#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=piech
#SBATCH --account piech
#SBATCH --mem=10G

conda activate offline_rl

python sepsis_analysis.py --scorer WIS --compute_true_mse --n 100 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 200 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 300 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 400 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 500 --env mdp

python sepsis_analysis.py --scorer WIS --compute_true_mse --n 600 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 700 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 800 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 900 --env mdp
python sepsis_analysis.py --scorer WIS --compute_true_mse --n 1000 --env mdp