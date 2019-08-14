#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 \
    python -m pdb tools/test_voc_eval.py \
        config/faster_rcnn_vgg_voc07.py  \
        /home/zzy/Projects/detector/work_dirs/faster_rcnn_vgg_voc07_news/epoch_6.pth output \
        --eval