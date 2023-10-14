import os
import json

import torch

from models import Autoencoder, ConvolutionalAutoencoder, ConvolutionalVAE, VariationalAutoencoder
from utils.dataloader import get_dataloader
from utils.trainer import train_autoencoder, visualize_reconstructions, load_checkpoint, evaluate_autoencoder
from settings import settings


def load_params(path):
    with open(path, "r", encoding='utf-8') as file:
        params = json.load(file)
    return params


def main(load_trained_model):
    params = load_params(settings.PATH_PARAMS_JSON)

    batch_size = params["batch_size"]
    resolution = params["resolution"]
    encoding_dim = params["encoding_dim"]
    num_epochs = params["num_epochs"]
    learning_rate = params.get("learning_rate", 0.001)
    ae_type = params["ae_type"]

    # Calculate input_dim based on resolution
    input_dim = 3 * resolution * resolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(settings.DATA_PATH, batch_size)

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

    if not load_trained_model:
        train_autoencoder(
            model,
            dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            start_epoch=start_epoch,
            optimizer=optimizer,
            ae_type=ae_type
        )
        print(f"Training complete up to epoch {num_epochs}!")

    valid_dataloader = get_dataloader(settings.VALID_DATA_PATH, batch_size)
    avg_valid_loss = evaluate_autoencoder(model, valid_dataloader, device, ae_type)
    print(f"Average validation loss: {avg_valid_loss:.4f}")

    visualize_reconstructions(
        model, valid_dataloader, num_samples=10, device=device, ae_type=ae_type
    )


if __name__ == "__main__":
    main(False)
