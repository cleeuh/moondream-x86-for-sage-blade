FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      build-essential \
      cmake \
      git \
      ninja-build \
      libopenblas-dev \
      libomp-dev \
      libjpeg-dev \
      libpng-dev \
      git \
      curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 \
      python3.10-distutils \
      python3.10-venv \
      python3.10-dev && \
    rm -rf /var/lib/apt/lists/*

# RUN python3 --version && sleep 10

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install --upgrade pip setuptools wheel





# Clone PyTorch source (v2.7.0 released April 23 2025) :contentReference[oaicite:2]{index=2}
RUN git clone --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout tags/v2.7.0 -b build-2.7.0

# RUN bash -c 'pwd && sleep 10 && ls pytorch && sleep 100'

WORKDIR /app/pytorch

# Install Python package requirements :contentReference[oaicite:3]{index=3}
RUN pip install -r requirements.txt

# Disable Kineto, Gloo and distributed support to skip those modules
ENV USE_KINETO=0 \
    USE_DISTRIBUTED=0 \
    USE_GLOO=0 \
    USE_FLASH_ATTENTION=0 \
    USE_MEM_EFF_ATTENTION=0
#     USE_NCCL=0


# Enable CUDA & cuDNN in the build; set architectures to compile for common GPUs :contentReference[oaicite:4]{index=4}
ENV USE_CUDA=1 \
    USE_CUDNN=0 \
    CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
    CUDNN_INCLUDE_DIR=/usr/local/cuda/include \
    TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"

# Build and install PyTorch
# RUN python3 setup.py build && \
#     python3 setup.py install

RUN python3 setup.py build

WORKDIR /app
# COPY . .

RUN pip3 install -U pywaggle[all]

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt



WORKDIR /app/pytorch
RUN python3 setup.py install
WORKDIR /app



COPY ./download_model_v2.py ../

RUN cd .. && python3 download_model_v2.py
# COPY huggingface ../huggingface

COPY src .

# ENTRYPOINT ["python3", "main.py"]
ENTRYPOINT ["python3", "test_samples.py"]
# ENTRYPOINT ["bash -c 'sleep 100000'"]