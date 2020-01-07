#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 \
    python -m pdb tools/train.py  \
        config/faster_rcnn_vgg_voc07.py
