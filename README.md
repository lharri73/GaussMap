# Gauss Map

## Requirements
- cmake >=3.17
- pybind11 >=2.2 (installed with apt is easier)
- CUDA (tested on 11.2)

## Installation
Installation is simplified with python. 
Simply run `python setup.py install` and dependencies will be installed and configured. 

## Getting Started
1. Follow the install steps above to setup the python library.

1. Run CenterTrack on all splits of the nuscenes dataset. 

   These will be pickled together to allow for faster loading later. If you don't need all of them 
   (for example, only the test for evaluation, then you can only run the test set)

1. From the root of this repository, run `./pySrc/main.py [nuscenes_version] [nuscenes_split] [nuscenes_root]`


## Running CenterTrack
A couple of changes have been made to centertrack to allow for it to run on NuScenes. 

These changes can be found [here](https://github.com/lharri73/CenterTrack-Nuscenes).

This requires PyTorch version 1.4.0, torchvision 0.5.0, and Cuda <= 10.2

1. Follow the setup steps in the [centertrack repository](https://github.com/xingyizhou/CenterTrack)

    These are mostly in readme/INSTALL.md

1. Run the `convert_nuscenes` script. You'll have to adjust the DATA_ROOT at the top of this file

1. Create an alias in the root of your nuscenes directory called `anotations` that points to the 
   directory containing the `.json` files just created.

1. Download the pretrained model called `nuScenes_3Ddetection_e140` listed in `readme/MODEL_ZOO.md`.
   Place this in the `models` folder.

1. Run the following script, adjusting `dataset_version` for each split of the dataset you need
   ```bash
   python test.py ddd --exp_id nusc_det_full --load_model ../models/nuScenes_3Ddetection_e140.pth --dataset nuscenes --dataset_version mini-train
   ```
   and copy the `.json` file with the results in the `exp/ddd/$exp_id$` folder to one of `results/CenterTrack/{train,val,mini-train,mini-val,test}`.