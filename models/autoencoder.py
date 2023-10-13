import torch
import torch.nn as nn
import torch.nn.functional as F


# Autoencoder Linear
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Autoencoder Convolucional
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
        x, idxs1 = self.pool(F.relu(self.enc1(x)))
        x, idxs2 = self.pool(F.relu(self.enc2(x)))
        x, idxs3 = self.pool(F.relu(self.enc3(x)))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x


# Variational Autoencoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, encoding_dim=128):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(3 * 64 * 64, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, encoding_dim)

        # Latent space
        self.fc_mu = nn.Linear(encoding_dim, encoding_dim)
        self.fc_log_var = nn.Linear(encoding_dim, encoding_dim)

        # Decoder
        self.dec1 = nn.Linear(encoding_dim, encoding_dim)
        self.dec2 = nn.Linear(encoding_dim, 256)
        self.dec3 = nn.Linear(256, 512)
        self.dec4 = nn.Linear(512, 3 * 64 * 64)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)

        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = torch.sigmoid(self.dec4(x))

        return x, mu, log_var


# Convolucional Variational Autoencoder
class ConvolutionalVAE(nn.Module):
    def __init__(self):
        super(ConvolutionalVAE, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc_mu = nn.Linear(16 * 8 * 8, 128)
        self.fc_log_var = nn.Linear(16 * 8 * 8, 128)

        # Decoder
        self.decoder_input = nn.Linear(128, 16 * 8 * 8)
        self.dec1 = nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)

        # Decoding
        x = self.decoder_input(z)
        x = x.view(x.size(0), 16, 8, 8)  # Unflatten
        x = self.upsample(x)
        x = F.relu(self.dec1(x))
        x = self.upsample(x)
        x = F.relu(self.dec2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec3(x))

        return x, mu, log_var