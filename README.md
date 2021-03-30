# Gauss Map

## Requirements
- cmake >=3.17
- pybind11 >=2.2 (installed with apt is easier as `python3-pybind11`)
- yaml-cpp (apt package `libyaml-cpp-dev`)
- CUDA (tested on 10.2 & 11.2)

## Installation
There are two methods you can use: From source or from a docker container. To
jump in and get started, the Docker container is already set up and will require
minimal effort to get running. 

## Getting Started

### Docker environment
Using the Docker container provided allows for an easy way to access and modify
code without worrying about system libraries. You'll just need to get the Nvidia
container oauth token before you'll be able to pull a cuda image from Nvidia. 

1. Obtain the Nvidia oauth token
   To do this, go to [this
   page](https://docs.nvidia.com/dgx/ngc-registry-for-dgx-user-guide) and follow
   the steps under the *Get a New API Key for the NGC Registry* section. In step 5,
   copy the oauth token, **Don't lose it!**

1. Login to nvcr.io
   I've added a function in the run script to do this for you, this will login
   to nvcr.io and begin building the container. This will require ~3GB for the
   nvidia container. In this repository, run 
   ```bash
   ./run.sh --build --login <your oauth token>
   ```

Normally, when you run the container, you'll just need to run it with:
   ```bash
   ./run.sh --dataset_dir <path to nuscenes root>
   ```

### From source
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
