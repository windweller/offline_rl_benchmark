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

python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 100
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 200
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 300
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 400
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 500

python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 600
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 700
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 800
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 900
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1000

python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1100
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1200
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1300
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1400
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1500

python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 1600
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 2400
python sepsis_analysis.py --scorer IS --compute_bootstrap_mse --n 3200