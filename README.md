# Autoencoder Project

A simple implementation of an autoencoder using PyTorch.

This project aims to provide a basic framework for understanding, training, and evaluating autoencoders on any image size.

## Features

- Train various autoencoder architectures: standard, convolutional, variational, and convolutional variational autoencoder on your dataset of images.
- Visualize the reconstructions of the autoencoder.
- Evaluate the model on a separate validation set.
- Checkpointing: Ability to save and resume training from checkpoints.
- Save and load trained model weights.
- Utilize custom datasets by simply pointing to your directory.
- Test all available autoencoder architectures with a single command.
- Easily containerize and run using Docker.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL
- matplotlib
- Docker (if you want to containerize the application)

### Installation

#### Traditional Setup:

1. Clone the repository.
2. Navigate to the project directory and install the required libraries.

#### Using Docker:

1. Clone the repository.
2. Navigate to the project directory.
3. Build the Docker image:

```bash
docker build -t autoencoder_research .
```

4. Run the container:
```bash
docker run -it --rm -v $(pwd):/app autoencoder_research bash
```

**Note:** If you're using the Docker method, you'll be inside the container's shell after running the above command. You can execute Python scripts or any other commands just like you would on your local machine.

## Usage

1. Modify the `settings/settings.py` file to point to your training and validation dataset.
2. Modify the `json/params.json` file to reflect your training preferences.

### Training

To train the autoencoder as per your configurations, simply run:

```bash
python run.py
```

### Testing

To run a test routine for all the available autoencoder architectures, use:

```bash
python run.py --test
```

This will train each autoencoder for a few epochs and provide an overview of their performances.

## Common Issues and Solutions

### Issue: Docker Volume and Uppercase Paths on Windows

**Description:** On Windows, when trying to mount a local volume in Docker, an error might arise if the path contains uppercase letters. This is due to Docker expecting repository names (or paths) to be in lowercase.

**Solution:** Convert the current directory path to lowercase and replace backslashes with regular slashes. If you're using Git Bash or a similar terminal on Windows, follow the steps below:

1. Get the full path of the current directory in Windows format:

```bash
win_path=$(pwd -W)
```

2. Convert the path to lowercase and replace backslashes with regular slashes:

```bash
lowercase_path=$(echo $win_path | tr '[:upper:]' '[:lower:]' | sed 's|\\|/|g')
```

3. Use this path in your Docker command:

```bash
docker run -it --rm -v "/$lowercase_path":/app autoencoder_research bash
```
