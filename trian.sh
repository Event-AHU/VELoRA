#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29505 \
  /media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/yanghaoxiang/VELoRA/train.py \
  MARS \
  --batchsize 4