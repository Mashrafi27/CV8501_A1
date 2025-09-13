#!/bin/sh


#SBATCH -p nvidia
#SBATCH --gres=gpu:1

python -m src.train_late_fusion --cfg config/adni.yaml --epochs 20 --batch 2 --lr 1e-4 --dropout 0.2 --pretrained_mri