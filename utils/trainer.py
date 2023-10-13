import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


def train_autoencoder(model, dataloader, num_epochs=5, learning_rate=0.001, device='cpu', start_epoch=0, optimizer=None):
    criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, num_epochs):
        for data in dataloader:
            img = data.to(device)
            img = img.view(img.size(0), -1)
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        save_checkpoint(model, optimizer, epoch, './autoencoder_checkpoint.pth')

    return model


def visualize_reconstructions(model, dataloader, num_samples=10, device='cpu', save_path="./samples"):
    model.eval()
    samples = next(iter(dataloader))
    samples = samples[:num_samples].to(device)
    samples = samples.view(samples.size(0), -1)
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


def evaluate_autoencoder(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data in dataloader:
            img = data.to(device)
            img = img.view(img.size(0), -1)
            output = model(img)
            loss = criterion(output, img)
            total_loss += loss.item()
    return total_loss / len(dataloader)
