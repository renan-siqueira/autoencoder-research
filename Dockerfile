# Use a base image. In this case, we will use Python 3.11
FROM python:3.11-slim-buster

# Define the working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch GPU-specific dependencies
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
