
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

Training & testing:
```bash
cd ATM-TCR
./train_test.sh
```