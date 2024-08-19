#!/bin/bash
#SBATCH -A FUAL8_NWPAV
#SBATCH -p boost_fua_prod
#SBATCH --time 05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#sBATCH --gres=gpu:4
#SBATCH --mem=123000
#SBATCH --job-name=job_test

srun python vae_main.py
