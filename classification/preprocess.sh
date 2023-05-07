#!/usr/bin/bash
LOG_DIR=logs/swd
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python3 classification/preprocess_data.py --config='classification/preprocess_train.json' \
--logdir=${LOG_DIR} \
--data_path="dataset/modelnet40_ply_hdf5_2048/train/"

CUDA_VISIBLE_DEVICES=${GPU} python3 classification/preprocess_data.py --config='classification/preprocess_test.json' \
--logdir=${LOG_DIR} \
--data_path="dataset/modelnet40_ply_hdf5_2048/test/"
