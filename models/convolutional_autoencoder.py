import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def forward(self, x):
        x, _ = self.pool(F.relu(self.enc1(x)))
        x, _ = self.pool(F.relu(self.enc2(x)))
        x, _ = self.pool(F.relu(self.enc3(x)))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x
