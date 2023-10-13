import os
import torch

from models.autoencoder import Autoencoder
from utils.dataloader import get_dataloader
from utils.trainer import train_autoencoder, visualize_reconstructions, save_checkpoint, load_checkpoint, evaluate_autoencoder
from settings import settings


def main(load_trained_model):
    BATCH_SIZE = 32
    INPUT_DIM = 3 * 64 * 64
    ENCODING_DIM = 64
    NUM_EPOCHS = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(settings.DATA_PATH, BATCH_SIZE)

    model = Autoencoder(INPUT_DIM, ENCODING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 0
    if os.path.exists(settings.PATH_SAVED_MODEL):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, settings.PATH_SAVED_MODEL, device)
        print(f"Loaded checkpoint and continuing training from epoch {start_epoch}.")

    if not load_trained_model:
        for epoch in range(start_epoch, NUM_EPOCHS):
            train_autoencoder(model, dataloader, device=device)
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] complete!")
            save_checkpoint(model, optimizer, epoch, settings.PATH_SAVED_MODEL)

    valid_dataloader = get_dataloader(settings.VALID_DATA_PATH, BATCH_SIZE)
    avg_valid_loss = evaluate_autoencoder(model, valid_dataloader, device)
    print(f"Average validation loss: {avg_valid_loss:.4f}")
    visualize_reconstructions(model, valid_dataloader, num_samples=10, device=device)


if __name__ == "__main__":
    main(False)
