FROM nvcr.io/nvidia/cuda:11.2.2-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget

RUN add-apt-repository ppa:deadsnakes/ppa

## add the latest cmake version
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
        
RUN apt-get update && \
    apt-get install -y \
        python3.8 \
        python3-pybind11 \
        libyaml-cpp-dev \
        cmake=3.20.0-0kitware1 \
        python3-pip

RUN bash -c 'mkdir -p /opt/gaussMap/{config,include,pySrc,scripts,src,results}/'
RUN echo "alias python='python3.8'" >> ~/.bashrc
RUN echo "alias pip='pip3'" >> ~/.bashrc
COPY requirements.txt CMakeLists.txt setup.py README.md /opt/gaussMap/

WORKDIR /opt/gaussMap

