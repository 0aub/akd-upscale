FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir diffusers transformers accelerate matplotlib

WORKDIR /app
COPY . /app