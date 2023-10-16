# Start with an NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Define the working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install basic tools and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip

# Install project dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch GPU-specific dependencies
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
