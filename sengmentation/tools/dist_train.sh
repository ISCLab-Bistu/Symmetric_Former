#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500  tools/train.py configs/sem_fpn/sy_512x512_160k_ade20k.py --launcher pytorch --work-dir log/train/exp1 --options model.pretrained=models/ckpt_epoch_499.pth