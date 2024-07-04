#!/bin/bash


source ~/anaconda3/etc/profile.d/conda.sh
conda activate mamba

CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg configs/ego4d/QueryMAMBA.yaml \
  --opts \
    NOTE base \
    MODEL.N_LAYER 4 \
    MODEL.N_DEC_LAYER 4 \
    MODEL.D_MODEL 1024 \
    MODEL.SHARE_CLASSIFIER False \
    MODEL.ACTION_CLS False \
    MODEL.VERB_CLS True \
    MODEL.NOUN_CLS True \
    MODEL.DROPOUT 0. \
    MODEL.DROP_CLS 0.5 \
    MODEL.PAST_CLS True \
    DATA.DATA_ROOT_PATH /home/zhong/Documents/datasets/ego4d \
    DATA.VERSION 1 \
    DATA.TAU_O 94. \
    DATA.LONG_MEMORY_LENGTH 64. \
    USE_WANDB True \
    TRAIN.EPOCHS 15 \
    TRAIN.OPTIMIZER adamw \
    TRAIN.BATCH_SIZE 128 \
    TRAIN.LR 0.0001 \
    TRAIN.GRADIENT_CLIPPING 1.
    TRAIN.USE_MIXUP True \
    TRAIN.NUM_WORKERS 4 \
    VAL.NUM_WORKERS 4 \
    VAL.BATCH_SIZE 128 \
    DTYPE bfloat16 \
