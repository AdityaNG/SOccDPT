#! /bin/bash

python3.9 -m SOccDPT.scripts.eval_SOccDPT \
    --version 3 \
    --dataset bdd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_bdd/qmjmgfu1/checkpoint_epoch_15.pth

python3.9 -m SOccDPT.scripts.eval_SOccDPT \
    --version 3 \
    --dataset idd \
    --model_type dpt_swin2_tiny_256 \
    --load checkpoints_pretrained/SOccDPT_V3_dpt_swin2_tiny_256_idd/zyigujaa/checkpoint_epoch_15.pth

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type DPT_SwinV2_T_256

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type DPT_Hybrid

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type DPT_Large

# CUDA_VISIBLE_DEVICES="" python3.9 -m SOccDPT.scripts.eval_others \
#     --dataset bdd \
#     --device cpu \
#     --model_type monodepth2

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type monodepth2


python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type manydepth

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type zerodepth

python3.9 -m SOccDPT.scripts.eval_others \
    --dataset bdd \
    --model_type packnet
