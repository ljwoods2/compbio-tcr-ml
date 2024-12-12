
## Setup

Everything is done on an A100 GPU

```bash
module load mamba
module load cuda-11.7.0-gcc-11.2.0
mamba create -n atm-tcr python=3.8.10
source activate atm-tcr
pip install -r requirements.txt
cd ATM-TCR
```

## Train

```bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --infile /scratch/ljwoods2/data/BAP/epi_split/train.csv \
    --model_name epi_eplit 

CUDA_VISIBLE_DEVICES=0 python main.py \
    --infile /scratch/ljwoods2/data/BAP/tcr_split/train.csv \
    --model_name tcr_split
```

## Test

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --indepfile /scratch/ljwoods2/data/BAP/epi_split/test.csv \
    --infile /scratch/ljwoods2/data/BAP/epi_split/test.csv \
    --model_name epi_eplit.ckpt \
    --mode test

CUDA_VISIBLE_DEVICES=0 python main.py \
    --indepfile /scratch/ljwoods2/data/BAP/tcr_split/test.csv \
    --infile /scratch/ljwoods2/data/BAP/tcr_split/test.csv \
    --model_name tcr_split.ckpt \
    --mode test

```

Combine data for modified epi split
```bash
cat provided_data/train.csv fold0/train.csv fold1/train.csv \
    fold2/train.csv fold3/train.csv \
    fold4/train.csv fold5/train.csv \
    fold6/train.csv fold7/train.csv \
    fold8/train.csv fold9/train.csv > train.csv

cat provided_data/test.csv fold0/test.csv fold1/test.csv \
    fold2/test.csv fold3/test.csv \
    fold4/test.csv fold5/test.csv \
    fold6/test.csv fold7/test.csv \
    fold8/test.csv fold9/test.csv > test.csv
```