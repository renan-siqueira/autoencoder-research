import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalVAE(nn.Module):
    def __init__(self):
        super(ConvolutionalVAE, self).__init__()

        self.model_structure = 'convolutional'
        self.model_variant = 'vae'

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