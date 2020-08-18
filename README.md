# DVBE
This is an implementation for [Domain-aware Visual Bias Eliminating for Generalized Zero-Shot Learning](https://arxiv.org/pdf/2003.13261), which has been accepted by CVPR2020.

DVBE is a new state-of-the-art method for generalized zero-shot learning.

## Introduction
This project is a basic implementation of DVBE by pytorch platform.

To do:
1. adding the autoML part.
2. publishing the segmentation version.

## Requirements

1. Python 3.6

2. Pytorch 0.4.0

3. CUDA 8.0


## Implementations


## Datasets Prepare

1. Downloading correspond dataset, e.g., CUB, AWA2, aPY, and SUN. Assume your data path is ${PATH}. A provided url is: (https://pan.baidu.com/s/1RYCZzKOuhDObuO-l-Ig78A     73rw)

2. Changing the data path around the line 190 of main.py, according to your ${PATH}.

## Two-stage Training
The training examples for the four datasets have been given in ./cub, ./awa2, ./apy, and ./sun.

In details, the training processing of DVBE consists of two stages, which is:

1. Run `train.py` to train DVBE with fixed backbone

	e.g. for training CUB 

	```shell
	python main.py -a dvbe -d cub -s /output --backbone resnet101 -b 128 
					--pretrained --is_fix
	```

2. Finetune the whole DSEN

	e.g. for training CUB
	```shell
	python main.py -a dvbe -d cub -s /output --backbone resnet101
					-b 16 --lr 0.001 \
					--epoch 180 --resume ./checkpoints/fix.model
	```
For reproducibility, a suggested seed can be cub: (5181,6803), awa2: (142,6059), apy: (119,4).
Besies, better performance can be obtained by flipping test, when setting --args.val_flippingtest

The reimplementation results and models are soon provided!
