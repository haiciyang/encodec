#!/bin/bash

#export CUDA_VISIBLE_DEVICES=6
#export WORLD_SIZE=2
#export RANK=1
#export LOCAL_RANK=1
#export NODE_RANK=1
#export MASTER_PORT=8890
#export MASTER_ADDR=transformer
#PATH=/home/anakuzne/miniconda3/envs/encodec/bin:$PATH

CUDA_VISIBLE_DEVICES=6 python -m encodec.dist_train --use_disc --disc_freq 1 --exp_name encodec_base_ft_3e5 --finetune_model /home/anakuzne/projects/encodec_hy/encodec/ckpt/epoch1999_model.amlt