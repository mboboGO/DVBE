MODEL=d2ve
DATA=cub
BACKBONE=resnet101

nvidia-smi

python eval.py -a ${MODEL} -d ${DATA} --backbone ${BACKBONE} -b 12 --resume /output/d2ve_0.6852.model

