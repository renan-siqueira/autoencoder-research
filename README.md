# Autoencoder Project

A simple implementation of an autoencoder using PyTorch. 

This project aims to provide a foundational structure to understand, train, and evaluate autoencoders on 64x64 images.

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

1. Clone the repository:

```bash
git clone https://github.com/renan-siqueira/autoencoder-project.git
```
2. Navigate to the project directory and install the required libraries:

```bash
cd autoencoder-project
pip install -r requirements.txt
```

## Usage

1. Modify settings/settings.py to point to your training and validation dataset.
2. To train the autoencoder, simply run:

```bash
python run.py
```

By default, this will train a new model. If you wish to use a pre-trained model, modify the `main` method in `run.py`.
