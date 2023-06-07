#!/bin/bash
#SBATCH -J scvae_wot
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --mem=30G
#SBATCH -o res_wot.out
#SBATCH -e err_wot.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=sanyu_rajakumar@brown.edu
module load anaconda/3-5.2.0 cudnn cuda/11.1.1 gcc/10.2
source activate ~/scratch/anaconda/scVAE
python /users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Models/scvae/FullRun.py -stu WOT -lhsdt 2023-04-22-at-13-38-21 -ttp 15 
