# Autoencoder Project

A simple implementation of an autoencoder using PyTorch. 

This project aims to provide a basic framework for understanding, training and evaluating autoencoders on any image size.

## Features

- Train various autoencoder architectures: standard, convolutional, variational, and convolutional variational autoencoder on your dataset of images.
- Visualize the reconstructions of the autoencoder.
- Evaluate the model on a separate validation set.
- Checkpointing: Ability to save and resume training from checkpoints.
- Save and load trained model weights.
- Utilize custom datasets by simply pointing to your directory.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL
- matplotlib

### Installation

1. Clone the repository.
2. Navigate to the project directory and install the required libraries.

## Usage

1. Modify the `settings/settings.py` file to point to your training and validation dataset.
2. Modify the `json/params.json` file to reflext your training preferences. 
3. To train the autoencoder, simply run:

```bash
python run.py
```
