#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500  tools/test.py configs/swin/mask_rcnn_sy_full_patch4_window7_mstrain_480-800_adamw_1x_coco.py log/train/exp1/epoch_4.pth --eval "bbox" --launcher pytorch