import torch.nn as nn
import torch


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        print('***** Denoising Autoencoder input_dim:', input_dim)
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        noise = torch.randn_like(x) * 0.1
        x_corrupted = x + noise

        x_encoded = self.encoder(x_corrupted)
        x_decoded = self.decoder(x_encoded)

        return x_decoded
