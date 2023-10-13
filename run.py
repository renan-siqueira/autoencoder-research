import os
import torch

from models.autoencoder import Autoencoder, ConvolutionalAutoencoder, VariationalAutoencoder, ConvolutionalVAE
from utils.dataloader import get_dataloader
from utils.trainer import train_autoencoder, visualize_reconstructions, load_checkpoint, evaluate_autoencoder
from settings import settings


def main(load_trained_model, ae_type='ae'):
    BATCH_SIZE = 32
    INPUT_DIM = 3 * 64 * 64
    ENCODING_DIM = 64
    NUM_EPOCHS = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(settings.DATA_PATH, BATCH_SIZE)

    if ae_type == 'ae':
        model = Autoencoder(INPUT_DIM, ENCODING_DIM).to(device)
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
            num_epochs=NUM_EPOCHS,
            device=device,
            start_epoch=start_epoch,
            optimizer=optimizer,
            ae_type=ae_type
        )
        print(f"Training complete up to epoch {NUM_EPOCHS}!")

    valid_dataloader = get_dataloader(settings.VALID_DATA_PATH, BATCH_SIZE)
    avg_valid_loss = evaluate_autoencoder(model, valid_dataloader, device, ae_type)
    print(f"Average validation loss: {avg_valid_loss:.4f}")
    
    visualize_reconstructions(
        model, valid_dataloader, num_samples=10, device=device, ae_type=ae_type
    )


if __name__ == "__main__":
    main(False, ae_type='conv_vae')
