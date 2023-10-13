import os
import torch

from models.autoencoder import Autoencoder
from utils.dataloader import get_dataloader
from utils.trainer import train_autoencoder, visualize_reconstructions, save_model, load_model, evaluate_autoencoder
from settings import settings


def main(load_trained_model):
    BATCH_SIZE = 32
    INPUT_DIM = 3 * 64 * 64
    ENCODING_DIM = 12
    NUM_EPOCHS = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(settings.DATA_PATH, BATCH_SIZE)
    model = Autoencoder(INPUT_DIM, ENCODING_DIM).to(device)

    if load_trained_model:
        trained_model = load_model(model, settings.PATH_SAVED_MODEL, device=device)
    else:
        trained_model = train_autoencoder(model, dataloader, NUM_EPOCHS, device=device)

    valid_dataloader = get_dataloader(settings.VALID_DATA_PATH, BATCH_SIZE)

    save_path = os.path.join('./', settings.PATH_SAVED_MODEL)
    save_model(trained_model, save_path)
    print(f"Model saved to {save_path}")

    avg_valid_loss = evaluate_autoencoder(trained_model, valid_dataloader, device)
    print(f"Average validation loss: {avg_valid_loss:.4f}")

    visualize_reconstructions(trained_model, valid_dataloader, num_samples=10, device=device)


if __name__ == "__main__":
    main(False)
