FROM oligoformer:latest AS weights

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
    python3-rna \
    vienna-rna \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir \
    numpy pandas scipy scikit-learn biopython tqdm fair-esm matplotlib \
    ptflops yacs pyyaml pytorch-ignite

ENV TORCH_HOME=/opt/torch

RUN mkdir -p "${TORCH_HOME}/hub/checkpoints"
COPY --from=weights /opt/torch/hub/checkpoints/RNA-FM_pretrained.pth ${TORCH_HOME}/hub/checkpoints/RNA-FM_pretrained.pth

WORKDIR /work/competitors/tools/attsioff
