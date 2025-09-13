#!/bin/sh


#SBATCH -p nvidia
#SBATCH --gres=gpu:1

python -m src.train_mri --cfg config/adni.yaml --epochs 20 --batch 2 --lr 1e-4 --pretrained