#!/bin/bash
#SBATCH -J scvae_dros
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --mem=30G
#SBATCH -o res_dros.out
#SBATCH -e err_dros.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=yuqi_lei@brown.edu
module load anaconda/3-4.3.0 cudnn cuda/11.1.1 gcc/10.2
source activate ~/scratch/anaconda/scVAE
python FullRun.py -stu drosophila -lhsdt 2023-04-18-at-06-46-12 -ttp 9 -stu drosophila --save_dir ../../Output/drosophila/scvae --data_dir ../../Data 

