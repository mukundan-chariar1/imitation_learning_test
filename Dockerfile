# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    wget \
    unzip \
    git \
    python3-pip \
    python3-dev \
    # Add CUDA development tools if needed
    cuda-command-line-tools-11-8 \
    libcudnn8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create Conda environment
RUN conda create -n irl-project python=3.10 -y

# Set up environment path
ENV CONDA_ENV_PATH="/opt/conda/envs/irl-project"
ENV PATH="$CONDA_ENV_PATH/bin:$PATH"

# Create a directory for potential volume mounting from host
WORKDIR /app

# Set CUDA environment variables more generically to work in different environments
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create a more robust entrypoint script
RUN echo '#!/bin/bash \n\
# Install requirements if present \n\
if [ -f requirements.txt ]; then \n\
    ${CONDA_ENV_PATH}/bin/pip install -r requirements.txt \n\
fi \n\
\n\
# Check for NVIDIA driver/GPU availability \n\
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then \n\
    echo "GPU detected, using GPU for JAX" \n\
    # Ensure latest nvidia-container-runtime is being used \n\
    export NVIDIA_VISIBLE_DEVICES=all \n\
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility \n\
else \n\
    echo "No GPU detected or NVIDIA drivers not working, forcing JAX to use CPU" \n\
    export JAX_PLATFORMS=cpu \n\
fi \n\
\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]