#!/bin/bash
#SBATCH -J scvae_pbmc_tuning
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 14:00:00
#SBATCH --mem=80G
#SBATCH -o res_pbmc_tuning.out
#SBATCH -e err_pbmc_tuning.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=sanyu_rajakumar@brown.edu
module load anaconda/3-5.2.0 cudnn cuda/11.1.1 gcc/10.2
source activate ~/scratch/anaconda/scVAE
python /users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Models/scvae/tune.py -stu PBMC -ttp 0 -tuti 720 --tune_all 
# latent size upper bound is 64 be default
# not tuning anything, checking if default GMVAE works
