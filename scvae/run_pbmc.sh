#!/bin/bash
#SBATCH -J scvae_pbmc
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --mem=30G
#SBATCH -o res_pbmc.out
#SBATCH -e err_pbmc.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=sanyu_rajakumar@brown.edu
module load anaconda/3-5.2.0 cudnn cuda/11.1.1 gcc/10.2
source activate ~/scratch/anaconda/scVAE
python /users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Models/scvae/FullRun.py -stu PBMC -lhsdt 2023-05-08-at-22-04-09 -ttp 0
