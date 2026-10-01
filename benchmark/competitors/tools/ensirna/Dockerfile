FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    python3-rdkit \
    pkg-config \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG BUILD_JOBS=4
COPY checkpoints/ViennaRNA-2.6.4.tar.gz /tmp/ViennaRNA-2.6.4.tar.gz
RUN cd /tmp && tar -xzf ViennaRNA-2.6.4.tar.gz && cd ViennaRNA-2.6.4 \
    && ./configure --without-perl --without-python --without-python3 --without-doc --without-tutorial --disable-unittests \
    && make -j"${BUILD_JOBS}" && make install && ldconfig \
    && rm -rf /tmp/ViennaRNA-2.6.4 /tmp/ViennaRNA-2.6.4.tar.gz

RUN python3 -m pip install --no-cache-dir --upgrade pip

# Package versions observed in the benchmark training image.
COPY requirements.frozen.txt /opt/sirbench-requirements.frozen.txt
ENV PIP_CONSTRAINT=/opt/sirbench-requirements.frozen.txt

RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir \
    numpy pandas scipy scikit-learn tqdm openpyxl biopython fair-esm \
    tensorboard xgboost rna-fm ptflops yacs pyyaml pytorch-ignite requests ViennaRNA==2.6.4

ENV TORCH_HOME=/opt/torch

RUN mkdir -p "${TORCH_HOME}/hub/checkpoints"
COPY checkpoints/RNA-FM_pretrained.pth ${TORCH_HOME}/hub/checkpoints/RNA-FM_pretrained.pth

WORKDIR /work/competitors/tools/ensirna
