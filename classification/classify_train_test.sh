#!/usr/bin/bash
LOG_DIR=logs/swd
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python3 classification/classification_train.py --config='classification/class_train_config.json' \
--logdir=${LOG_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python3 classification/classification_test.py --config='classification/class_test_config.json' \
--logdir=${LOG_DIR}
