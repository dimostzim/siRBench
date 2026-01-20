FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir \
    numpy pandas transformers matplotlib

ENV BERT_DIR=/opt/dnabert/6mer

RUN python3 - <<'PY'
from transformers import AutoModel, AutoTokenizer
import os

model_id = "zhihan1996/DNA_bert_6"
target = os.environ.get("BERT_DIR", "/opt/dnabert/6mer")
os.makedirs(target, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(target)
model = AutoModel.from_pretrained(model_id)
model.save_pretrained(target)
PY

WORKDIR /work/competitors/tools/sirnabert
