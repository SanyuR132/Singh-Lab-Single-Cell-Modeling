#!/bin/bash
#SBATCH -J scvae_dros_tuning
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 14:00:00
#SBATCH --mem=80G
#SBATCH -o res_dros_tuning.out
#SBATCH -e err_dros_tuning.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=yuqi_lei@brown.edu
module load anaconda/3-5.2.0 cudnn cuda/11.1.1 gcc/10.2
source activate scvae
which python
~/anaconda/scvae/bin/python ./tune.py -stu drosophila -ttp 3 -tuti 10 --tune_all --save_dir ../../Output/drosophila/scvae --data_dir ../../Data 

# latent size upper bound is 64 be default
# not tuning anything, checking if default GMVAE works
