FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ARG VIENNA_VERSION=2.4.18

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libgsl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/ViennaRNA/ViennaRNA/releases/download/v${VIENNA_VERSION}/ViennaRNA-${VIENNA_VERSION}.tar.gz \
    && tar -xzf ViennaRNA-${VIENNA_VERSION}.tar.gz \
    && cd ViennaRNA-${VIENNA_VERSION} \
    && ./configure --without-perl --without-python --without-forester \
    && make -j"$(nproc)" \
    && make install \
    && cd .. \
    && rm -rf ViennaRNA-${VIENNA_VERSION} ViennaRNA-${VIENNA_VERSION}.tar.gz \
    && ldconfig

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    pandas scikit-learn biopython chardet \
    tensorflow==2.4.1 stellargraph==1.2.1

WORKDIR /work/competitors/tools/gnn4sirna
