#### Table of Content

- [Self-Attention Amortized Distributional Projection Optimization for Sliced Wasserstein Point-Cloud Reconstruction](#self-attention-amortized-distributional-projection-optimization-for-sliced-wasserstein-point-cloud-reconstruction)
  - [Getting Started](#getting-started)
    - [Datasets](#datasets)
      - [ShapeNet Core with 55 categories (refered from FoldingNet.)](#shapenet-core-with-55-categories-refered-from-foldingnet)
      - [ModelNet40](#modelnet40)
      - [ShapeNet Chair](#shapenet-chair)
      - [3DMatch](#3dmatch)
    - [Installation](#installation)
    - [For docker users](#for-docker-users)
  - [Experiments](#experiments)
    - [Point-cloud reconstruction](#point-cloud-reconstruction)
    - [Amortization gap](#amortization-gap)
    - [Transfer learning](#transfer-learning)
    - [Point-cloud generation](#point-cloud-generation)
  - [Acknowledgment](#acknowledgment)


# Self-Attention Amortized Distributional Projection Optimization for Sliced Wasserstein Point-Cloud Reconstruction

This repository contains the official Python3 implementation of our ICML paper "**Self-Attention Amortized Distributional Projection Optimization for Sliced Wasserstein Point-Cloud Reconstruction**". 

Details of the model architecture and experimental settings can be found in our paper.

```bibtex
@InProceedings{nguyen2023self,
    title={Self-Attention Amortized Distributional Projection Optimization for Sliced Wasserstein Point-Cloud Reconstruction}, 
    author={Khai Nguyen and Dang Nguyen and Nhat Ho},
    booktitle={Proceedings of the 40th International Conference on Machine Learning},
    year={2023},
}
```

Please CITE our papers whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by Dang Nguyen and Khai Nguyen. README is on updating process.

## Getting Started
### Datasets
#### ShapeNet Core with 55 categories (refered from <a href="http://www.merl.com/research/license#FoldingNet" target="_blank">FoldingNet</a>.)
```bash
  cd dataset
  bash download_shapenet_core55_catagories.sh
```
#### ModelNet40
```bash
  cd dataset
  bash download_modelnet40_same_with_pointnet.sh
```
#### ShapeNet Chair
```bash
  cd dataset
  bash download_shapenet_chair.sh
``` 
#### 3DMatch
```bash
  cd dataset
  bash download_3dmatch.sh
```
### Installation
The code has been tested with Python 3.6.9, PyTorch 1.2.0, CUDA 10.0 on Ubuntu 18.04.  

To install the required python packages, run
```bash
pip install -r requirements.txt
```

To compile CUDA kernel for CD/EMD loss:
```bash
cd metrics_from_point_flow/pytorch_structural_losses/
make clean
make
```

### For docker users
For building the docker image simply run the following command in the root directory
```bash
docker build -f Dockerfile -t <tag> .
```

## Experiments
### Point-cloud reconstruction
Available arguments for training an autoencoder
```bash
train.py [-h] [--config CONFIG] [--logdir LOGDIR]
                [--data_path DATA_PATH] [--loss LOSS]
                [--autoencoder AUTOENCODER]

optional arguments:
  -h, --help                  show this help message and exit
  --config CONFIG             path to json config file
  --logdir LOGDIR             path to the log directory
  --data_path DATA_PATH       path to data for training
  --loss LOSS                 loss function. One of [swd, emd, chamfer, asw, msw, vsw, amortized_msw, amortized_vsw]
  --autoencoder AUTOENCODER   model name. One of [pointnet, pcn]
  --f_type                    amortized model type [linear, glinear, nonlinear, attn, eff_attn, lin_attn]
  --inter_dim                 dimension of keys
  --proj_dim                  projected dimension in linformer
  --kappa                     scale of vMF distribution
```

Example
```bash
python3 train.py --config="config.json" \
                --logdir="logs/" \
                --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" \
                --loss="swd" \
                --autoencoder="pointnet"

# or in short, you can run
bash train.sh
```

To test reconstruction
```bash
python3 reconstruction/reconstruction_test.py  --config="reconstruction/config.json" \
                                              --logdir="logs/" \
                                              --data_path="dataset/modelnet40_ply_hdf5_2048/"

# or in short, you can run
bash reconstruction/test.sh
```

To render input and reconstructed point-clouds, please follow the instruction in `render/README.md` to install dependencies. To reproduce Figure 4 in our paper, run the following commands *after training all autoencoders*
```bash
python3 save_point_clouds.py
cd render
bash render_reconstruction.sh
cd ..
python3 concat_reconstructed_images.py
```

### Amortization gap
To run ablation study
```bash
python3 approximate_amortized_vsw.py
```

### Transfer learning
To generate latent codes of the train/test sets of ModelNet40 and save them into files
```bash
python3 classification/preprocess_data.py  --config='classification/preprocess_train.json' \
                                          --logdir="logs/" \
                                          --data_path="dataset/modelnet40_ply_hdf5_2048/train/"

python3 classification/preprocess_data.py  --config='classification/preprocess_test.json' \
                                          --logdir="logs/" \
                                          --data_path="dataset/modelnet40_ply_hdf5_2048/test/"

# or in short, you can run
bash classification/preprocess.sh
```

To get classification results
```bash
python3 classification/classification_train.py --config='classification/class_train_config.json' \
                                              --logdir="logs/"

python3 classification/classification_test.py  --config='classification/class_test_config.json' \
                                              --logdir="logs/"

# or in short, you can run
bash classification/classify_train_test.sh
```

### Point-cloud generation
To train the autoencoder, run
```bash
bash train_generation_ae.sh
```

To generate latent codes of train/test sets of ShapeNet Chair, then train and test the generator, simply run
```bash
bash generation/run_generation.sh
```

## Acknowledgment
The structure of this repo is largely based on [PointSWD](https://github.com/VinAIResearch/PointSWD). The structure of folder `render` is largely based on [Mitsuba2PointCloudRenderer](https://github.com/tolgabirdal/Mitsuba2PointCloudRenderer). We are very grateful for their open sources.