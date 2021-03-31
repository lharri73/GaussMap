FROM nvcr.io/nvidia/cuda:11.2.2-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget \
        vim \
        git

RUN add-apt-repository ppa:deadsnakes/ppa

## add the latest cmake version
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
        
RUN apt-get update && \
    apt-get install -y \
        python3.8 \
        libyaml-cpp-dev \
        cmake=3.20.0-0kitware1 \
        python3-pip \
        python3.8-dev

RUN bash -c 'mkdir -p /opt/gaussMap/{config,include,pySrc,scripts,src,results}/'
RUN ln -sf /usr/bin/python3.8 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN wget -O pybind11.deb http://archive.ubuntu.com/ubuntu/pool/universe/p/pybind11/python3-pybind11_2.5.0-5_all.deb && \
    wget -O pybind11-dev.deb http://archive.ubuntu.com/ubuntu/pool/universe/p/pybind11/pybind11-dev_2.5.0-5_all.deb && \
    apt-get install -y ./pybind11.deb ./pybind11-dev.deb && rm *.deb
COPY requirements.txt CMakeLists.txt setup.py README.md /opt/gaussMap/

WORKDIR /opt/gaussMap

