#!/bin/bash
cd ..
export PYTHONPATH="/userhome/codes/D2VE/vision/":$PYTHONPATH


MODEL=d2ve
DATA=apy
BACKBONE=resnet101
SAVE_PATH=./${DATA}/checkpoints/${MODEL}
mkdir -p ${SAVE_PATH}

python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 128 --lr1 0.001 --pretrained --is_fix &> ${SAVE_PATH}/fix.log

python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 12 --lr1 0.00005 --epochs 180 --resume ${SAVE_PATH}/fix.model &> ${SAVE_PATH}/ft.log

