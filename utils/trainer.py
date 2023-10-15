import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from settings import settings


def save_reconstructed_images(model, samples, epoch, device, save_dir="./training"):
    # Mova as amostras para o dispositivo e ajuste sua forma se necessário
    samples = samples.to(device)
    if model.model_structure == 'linear':
        samples = samples.view(samples.size(0), -1)

    # Gere as reconstruções
    if model.model_variant == 'vae':
        reconstructions, _, _ = model(samples)
    elif model.model_variant == 'sparse':
        reconstructions, _ = model(samples)
    else:
        reconstructions = model(samples)

    # Converta as reconstruções para a forma original (C x H x W)
    if model.model_structure == 'linear':
        reconstructions = reconstructions.view(-1, 3, int((samples.shape[1] / 3) ** 0.5), int((samples.shape[1] / 3) ** 0.5))

    # Salve as reconstruções
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_image(reconstructions, os.path.join(save_dir, f"reconstructed_epoch_{epoch}.png"))


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def save_checkpoint_file(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch + 1


def train_autoencoder(
        model, dataloader, num_epochs, learning_rate, device,
        start_epoch, optimizer, save_checkpoint, ae_type
    ):

    saved_samples = os.path.join(settings.PATH_SAVED_SAMPLES, ae_type)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, num_epochs + 1):
        for data in dataloader:
            img = data.to(device)

            if model.model_structure == 'linear':
                img = img.view(img.size(0), -1)

            if model.model_variant == 'vae':
                recon_x, mu, log_var = model(img)
                loss = loss_function_vae(recon_x, img, mu, log_var)
            elif model.model_variant == 'sparse':
                output, encoded = model(img)
                loss = loss_function_sparse(output, encoded, img)
            else:
                output = model(img)
                criterion = nn.MSELoss()
                loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()

            # Clip de gradientes para prevenir gradientes explosivos
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

        # Checar anomalias no valor da perda
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Anomaly detected in loss at epoch {epoch}. Finishing the training.")
            return model

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

        sample_data = next(iter(dataloader))
        save_reconstructed_images(model, sample_data, epoch, device, save_dir=saved_samples)

        if save_checkpoint:
            save_checkpoint_file(model, optimizer, epoch, settings.PATH_SAVED_MODEL)

    return model


def loss_function_vae(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def loss_function_sparse(output, encoded, target, sparsity_target=0.05, sparsity_weight=1e-3):
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(output, target)

    sparsity_loss = 0

    if len(encoded.shape) == 2:  # Sparse Linear Autoencoder
        avg_activation = torch.mean(encoded, dim=0)
    elif len(encoded.shape) == 4:  # Sparse Convolutional Autoencoder
        avg_activation = torch.mean(encoded, dim=(0, 2, 3))
    else:
        raise ValueError("Unexpected shape for encoded tensor")

    EPS = 1e-10
    avg_activation = torch.clamp(avg_activation, EPS, 1 - EPS)
    sparsity_loss += sparsity_target * torch.log(sparsity_target / avg_activation)
    sparsity_loss += (1 - sparsity_target) * torch.log((1 - sparsity_target) / (1 - avg_activation))
    sparsity_loss = torch.sum(sparsity_loss)

    total_loss = recon_loss + sparsity_weight * sparsity_loss

    return total_loss


def evaluate_autoencoder(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data in dataloader:
            img = data.to(device)

            if model.model_structure == 'linear':
                img = img.view(img.size(0), -1)

            if model.model_variant == 'vae':
                output, _, _ = model(img)
            elif model.model_variant == 'sparse':
                output, _ = model(img)
            else:
                output = model(img)
            loss = criterion(output, img)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def visualize_reconstructions(model, dataloader, num_samples=10, device='cpu', save_path="./samples", resolution=64):
    model.eval()
    samples = next(iter(dataloader))
    samples = samples[:num_samples].to(device)

    if model.model_structure == 'linear':
        samples = samples.view(samples.size(0), -1)

    if model.model_variant == 'vae':
        reconstructions, _, _ = model(samples)
    elif model.model_variant == 'sparse':
        reconstructions, _ = model(samples)
    else:
        reconstructions = model(samples)

    samples = samples.view(-1, 3, resolution, resolution)
    reconstructions = reconstructions.view(-1, 3, resolution, resolution)

    combined = torch.cat([samples, reconstructions], dim=0)
    grid_img = make_grid(combined, nrow=num_samples)

    plt.imshow(grid_img.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.show()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_image(grid_img, os.path.join(save_path, 'combined_samples.png'))
