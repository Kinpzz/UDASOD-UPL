#! /usr/bin/bash

## reusme training
CUDA_VISIBLE_DEVICES=1 python train.py \
    --exp_config 'exp_1/ldf_train/lr=0.05_bn=32_eps=36-7*36_pse=src_half_psrc=0.5_ptgt=0.1_sche=cyc-0.75_lbsec=fuse-1-1-1_aug=w/config.yaml' \
    --resume


## training from scratch
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_config 'config/ldf_train.yaml' \
    SEED 1218 \
    MODEL.WARMUP_PATH ""

## training from scratch for vgg backbone
CUDA_VISIBLE_DEVICES=3 python train.py \
    --exp_config 'config/ldf_vgg_train.yaml' \
    SEED 1218 \
    MODEL.WARMUP_PATH ""

## training from warmp ckpt
CUDA_VISIBLE_DEVICES=3 python train.py \
    --exp_config 'config/ldf_train.yaml' \
    MODEL.WARMUP_PATH <warmup_ckpt_path>
