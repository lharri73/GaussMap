# Gauss Map

## Requirements
- cmake >=3.17
- pybind11 >=2.2 (installed with apt is easier as `python3-pybind11`)
- yaml-cpp (apt package `libyaml-cpp-dev`)
- CUDA (tested on 10.2 & 11.2)

## Installation
Installation is simplified with python. 
Simply run `python setup.py install` and dependencies will be installed and configured. 

## Getting Started
1. This requires a newer version of cmake than is available with most Ubuntu
   versions. If on Ubuntu, a newer version can be installed with `apt`:
      1. `wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null`
      2. Add the repository to your sources list:
         - 20.04
           `sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'`
         - 18.04
           `sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'`
         - 16.04
           `sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'`
1. Run the following to install the python library
    ```bash
    python setup.py install
    ```

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
