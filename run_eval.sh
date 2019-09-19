#!/bin/bash
cd ./vision
python setup.py install
cd ..

export PYTHONPATH="/userhome/codes/VDR/vision/":$PYTHONPATH


MODEL=vden
DATA=cub
SAVE_PATH=./${DATA}/checkpoints/${MODEL}

mkdir -p ${SAVE_PATH}


python eval1.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} -b 128 -j 3  --resume ${SAVE_PATH}/vden_0.6842.model --is_fix

