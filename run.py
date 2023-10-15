import os
import json
import random
import time
import argparse

import numpy as np
import torch

from models import (
    Autoencoder,
    ConvolutionalAutoencoder,
    ConvolutionalVAE,
    DenoisingAutoencoder,
    SparseAutoencoder,
    VariationalAutoencoder,
    DenoisingConvolutionalAutoencoder,
    SparseConvolutionalAutoencoder
)

from settings import settings
from utils.dataloader import get_dataloader
from utils.trainer import train_autoencoder, visualize_reconstructions, load_checkpoint, evaluate_autoencoder
from utils import utils


def get_model_by_type(ae_type=None, input_dim=None, encoding_dim=None, device=None):
    models = {
        'ae': lambda: Autoencoder(input_dim, encoding_dim),
        'dae': lambda: DenoisingAutoencoder(input_dim, encoding_dim),
        'sparse': lambda: SparseAutoencoder(input_dim, encoding_dim),
        'vae': VariationalAutoencoder,
        'conv': ConvolutionalAutoencoder,
        'conv_dae': DenoisingConvolutionalAutoencoder,
        'conv_vae': ConvolutionalVAE,
        'conv_sparse': SparseConvolutionalAutoencoder,
    }

    if ae_type is None:
        return list(models.keys())

    if ae_type not in models:
        raise ValueError(f"Unknown AE type: {ae_type}")

    model = models[ae_type]()
    return model.to(device)


def set_seed(seed):
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


def main(load_trained_model, ae_type=None, num_epochs=5, test_mode=True):
    set_seed(1)
    params = load_params(settings.PATH_PARAMS_JSON)

    batch_size = params["batch_size"]
    resolution = params["resolution"]
    encoding_dim = params["encoding_dim"]
    learning_rate = params.get("learning_rate", 0.001)
    save_checkpoint = params["save_checkpoint"]

    if not ae_type:
        ae_type = params["ae_type"]
        num_epochs = params["num_epochs"]
        test_mode = False

    # Calculate input_dim based on resolution
    input_dim = 3 * resolution * resolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(settings.DATA_PATH, batch_size, resolution)

    model = get_model_by_type(ae_type, input_dim, encoding_dim, device)
    optimizer = torch.optim.Adam(model.parameters())

    try:
        if not load_trained_model:
            start_epoch = 1
            if os.path.exists(settings.PATH_SAVED_MODEL):
                model, optimizer, start_epoch = load_checkpoint(
                    model, optimizer, settings.PATH_SAVED_MODEL, device
                )
                print(f"Loaded checkpoint and continuing training from epoch {start_epoch}.")

            start_time = time.time()

            train_autoencoder(
                model,
                dataloader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                start_epoch=start_epoch,
                optimizer=optimizer,
                save_checkpoint=save_checkpoint,
                ae_type=ae_type
            )

            elapsed_time = utils.format_time(time.time() - start_time)
            print(f"\nTraining took {elapsed_time}")
            print(f"Training complete up to epoch {num_epochs}!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    if not test_mode:
        valid_dataloader = get_dataloader(settings.VALID_DATA_PATH, batch_size, resolution)
        avg_valid_loss = evaluate_autoencoder(model, valid_dataloader, device)
        print(f"\nAverage validation loss: {avg_valid_loss:.4f}\n")

        visualize_reconstructions(
            model, valid_dataloader, num_samples=10,
            device=device, resolution=resolution
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training and testing autoencoders.')
    parser.add_argument(
        '--test', action='store_true', help='Run the test routine for all autoencoders.'
    )

    args = parser.parse_args()

    if args.test:
        ae_types = get_model_by_type()
        for ae_type in ae_types:
            print(f"\n===== Training {ae_type} =====\n")
            main(load_trained_model=False, ae_type=ae_type)
    else:
        main(load_trained_model=False)
