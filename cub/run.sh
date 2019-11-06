#!/bin/bash

cd ..


MODEL=d2ve
DATA=cub
BACKBONE=resnet101
SAVE_PATH=./${DATA}/checkpoints/${MODEL}

mkdir -p ${SAVE_PATH}
nvidia-smi

python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 128 --pretrained --is_fix #&> ${SAVE_PATH}/fix.log

python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 12 --lr1 0.001 --epochs 180 --resume ${SAVE_PATH}/fix.model #&> ${SAVE_PATH}/ft.log
