import os
import json
import random
import time

import numpy as np
import torch

from models import Autoencoder, ConvolutionalAutoencoder, ConvolutionalVAE, VariationalAutoencoder
from settings import settings
from utils.dataloader import get_dataloader
from utils.trainer import train_autoencoder, visualize_reconstructions, load_checkpoint, evaluate_autoencoder
from utils import utils


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_params(path):
    with open(path, "r", encoding='utf-8') as file:
        params = json.load(file)
    return params


def main(load_trained_model):
    set_seed(1)
    params = load_params(settings.PATH_PARAMS_JSON)

    batch_size = params["batch_size"]
    resolution = params["resolution"]
    encoding_dim = params["encoding_dim"]
    num_epochs = params["num_epochs"]
    learning_rate = params.get("learning_rate", 0.001)
    ae_type = params["ae_type"]
    save_checkpoint = params["save_checkpoint"]

    # Calculate input_dim based on resolution
    input_dim = 3 * resolution * resolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(settings.DATA_PATH, batch_size, resolution)

    if ae_type == 'ae':
        model = Autoencoder(input_dim, encoding_dim).to(device)
    elif ae_type == 'conv':
        model = ConvolutionalAutoencoder().to(device)
    elif ae_type == 'vae':
        model = VariationalAutoencoder().to(device)
    elif ae_type == 'conv_vae':
        model = ConvolutionalVAE().to(device)
    else:
        raise ValueError(f"Unknown AE type: {ae_type}")

    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 0
    if os.path.exists(settings.PATH_SAVED_MODEL):
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, settings.PATH_SAVED_MODEL, device
        )
        print(f"Loaded checkpoint and continuing training from epoch {start_epoch}.")

    try:
        if not load_trained_model:
            start_time = time.time()

            train_autoencoder(
                model,
                dataloader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                start_epoch=start_epoch,
                optimizer=optimizer,
                ae_type=ae_type,
                save_checkpoint=save_checkpoint
            )

            elapsed_time = utils.format_time(time.time() - start_time)
            print(f"\nTraining took {elapsed_time}")
            print(f"Training complete up to epoch {num_epochs}!")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    valid_dataloader = get_dataloader(settings.VALID_DATA_PATH, batch_size, resolution)
    avg_valid_loss = evaluate_autoencoder(model, valid_dataloader, device, ae_type)
    print(f"\nAverage validation loss: {avg_valid_loss:.4f}\n")

    visualize_reconstructions(
        model, valid_dataloader, num_samples=10, device=device, ae_type=ae_type, resolution=resolution
    )


if __name__ == "__main__":
    main(False)
