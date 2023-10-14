import torch
import torch.nn as nn
import torch.nn.functional as F


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
