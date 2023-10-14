import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


def train_autoencoder(model, dataloader, num_epochs, learning_rate, device, start_epoch, optimizer, ae_type):
    criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, num_epochs):
        for data in dataloader:
            img = data.to(device)

            if ae_type not in ['conv', 'conv_vae']:
                img = img.view(img.size(0), -1)

            if ae_type in ['vae', 'conv_vae']:
                recon_x, mu, log_var = model(img)
                loss = loss_function_vae(recon_x, img, mu, log_var)
            else:
                output = model(img)
                loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        save_checkpoint(model, optimizer, epoch, './autoencoder_checkpoint.pth')

    return model


def loss_function_vae(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def evaluate_autoencoder(model, dataloader, device, ae_type):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data in dataloader:
            img = data.to(device)

            if ae_type not in ['conv', 'conv_vae']:
                img = img.view(img.size(0), -1)

            if ae_type in ['vae', 'conv_vae']:
                output, _, _ = model(img)
            else:
                output = model(img)
            loss = criterion(output, img)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def visualize_reconstructions(model, dataloader, num_samples=10, device='cpu', save_path="./samples", ae_type='ae'):
    model.eval()
    samples = next(iter(dataloader))
    samples = samples[:num_samples].to(device)

    if ae_type not in ['conv', 'conv_vae']:
        samples = samples.view(samples.size(0), -1)
    
    if ae_type in ['vae', 'conv_vae']:
        reconstructions, _, _ = model(samples)
    else:
        reconstructions = model(samples)

    samples = samples.view(-1, 3, 64, 64)
    reconstructions = reconstructions.view(-1, 3, 64, 64)

    combined = torch.cat([samples, reconstructions], dim=0)
    grid_img = make_grid(combined, nrow=num_samples)

    plt.imshow(grid_img.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.show()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_image(grid_img, os.path.join(save_path, 'combined_samples.png'))


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def save_checkpoint(model, optimizer, epoch, path):
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
