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
    --infile /scratch/ljwoods2/data/BAP/epi_split/train.csv \
    --model_name epi_eplit 

CUDA_VISIBLE_DEVICES=0 /home/ljwoods2/.conda/envs/atm-tcr/bin/python main.py \
    --infile /scratch/ljwoods2/data/BAP/tcr_split/train.csv \
    --model_name tcr_split

CUDA_VISIBLE_DEVICES=0 /home/ljwoods2/.conda/envs/atm-tcr/bin/python main.py \
    --indepfile /scratch/ljwoods2/data/BAP/epi_split/test.csv \
    --infile /scratch/ljwoods2/data/BAP/epi_split/test.csv \
    --model_name epi_eplit.ckpt \
    --mode test

CUDA_VISIBLE_DEVICES=0 /home/ljwoods2/.conda/envs/atm-tcr/bin/python main.py \
    --indepfile /scratch/ljwoods2/data/BAP/tcr_split/test.csv \
    --infile /scratch/ljwoods2/data/BAP/tcr_split/test.csv \
    --model_name tcr_split.ckpt \
    --mode test