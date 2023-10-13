import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from PIL import Image


def train_autoencoder(model, dataloader, num_epochs=5, learning_rate=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in dataloader:
            img = data.to(device)
            img = img.view(img.size(0), -1)
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def visualize_reconstructions(model, dataloader, num_samples=10, device='cpu', save_path="./samples"):
    model.eval()
    samples = next(iter(dataloader))
    samples = samples[:num_samples].to(device)
    samples = samples.view(samples.size(0), -1)
    reconstructions = model(samples)

    samples = samples.view(-1, 3, 64, 64)
    reconstructions = reconstructions.view(-1, 3, 64, 64)

    # Combine as amostras e reconstruções em uma única grade
    combined = torch.cat([samples, reconstructions], dim=0)
    grid_img = make_grid(combined, nrow=num_samples)

    # Visualização usando Matplotlib
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
