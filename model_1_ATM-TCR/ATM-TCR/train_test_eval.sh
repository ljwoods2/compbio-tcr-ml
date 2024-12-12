#!/bin/bash

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00
#SBATCH -p general
#SBATCH -o %j.out
#SBATCH --gres=gpu:a100:1

module purge
module load mamba
module load cuda-11.7.0-gcc-11.2.0
source activate atm-tcr

CUDA_VISIBLE_DEVICES=0 /home/ljwoods2/.conda/envs/atm-tcr/bin/python main.py \
    --indepfile /home/ljwoods2/workspace/compbio/ml/model_1_ATM-TCR/epitope_split_test.csv \
    --infile /home/ljwoods2/workspace/compbio/ml/model_1_ATM-TCR/epitope_split_test.csv \
    --model_name epi_eplit_modified.ckpt \
    --mode test

CUDA_VISIBLE_DEVICES=0 /home/ljwoods2/.conda/envs/atm-tcr/bin/python main.py \
    --indepfile /home/ljwoods2/workspace/compbio/ml/model_1_ATM-TCR/tcr_split_test.csv \
    --infile /home/ljwoods2/workspace/compbio/ml/model_1_ATM-TCR/tcr_split_test.csv \
    --model_name tcr_split.ckpt \
    --mode test