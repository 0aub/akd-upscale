FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir diffusers transformers accelerate matplotlib

WORKDIR /app
COPY . /app