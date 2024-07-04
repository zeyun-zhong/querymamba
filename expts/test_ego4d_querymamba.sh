#!/bin/bash


source ~/anaconda3/etc/profile.d/conda.sh
conda activate mamba

CUDA_VISIBLE_DEVICES=0 python main.py \
  --cfg configs/ego4d/QueryMAMBA.yaml \
  --opts \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TEST.CKPT_PATH YOUR_CHECKPOINT_PATH \
    VAL.NUM_WORKERS 4 \
    DATA.DATA_ROOT_PATH /home/zhong/Documents/datasets/ego4d \
    USE_WANDB False \
    MODEL.N_LAYER 4 \
    MODEL.N_DEC_LAYER 4 \
    VAL.BATCH_SIZE 128 \
    MODEL.D_MODEL 1024 \
    MODEL.SHARE_CLASSIFIER False \
    MODEL.ACTION_CLS False \
    MODEL.VERB_CLS True \
    MODEL.NOUN_CLS True \
    MODEL.PAST_CLS True
    DTYPE bfloat16 \
    DATA.TAU_O 94. \
    DATA.LONG_MEMORY_LENGTH 64. \
