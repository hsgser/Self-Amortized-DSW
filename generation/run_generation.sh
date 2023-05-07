#!/usr/bin/bash
GPU=0
SEED=1
LOG_DIR=generation_logs/chamfer

CUDA_VISIBLE_DEVICES=${GPU} taskset python3 generation/preprocess.py --config='generation/preprocess_train.json' --logdir=${LOG_DIR} --data_path="dataset/shapenet_chair/train.npz"

CUDA_VISIBLE_DEVICES=${GPU} taskset python3 generation/preprocess.py --config='generation/preprocess_test.json' --logdir=${LOG_DIR} --data_path="dataset/shapenet_chair/test.npz"

CUDA_VISIBLE_DEVICES=${GPU} taskset python3 generation/train_latent_generator.py --seed=${SEED} \
--logdir=${LOG_DIR}

CUDA_VISIBLE_DEVICES=${GPU} taskset python3 generation/test_generation.py --config='generation/test_generation_config.json' \
--logdir=${LOG_DIR}
