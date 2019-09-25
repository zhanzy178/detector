#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 \
    python -m pdb tools/test_voc.py \
        config/faster_rcnn_vgg_voc07.py  \
       work_dirs/faster_rcnn_vgg_voc07_4/latest.pth output \
        --eval