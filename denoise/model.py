import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 224x224 -> 224x224
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),  # 224x224 -> 112x112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 112x112 -> 112x112
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),  # 112x112 -> 56x56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 56x56 -> 56x56
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),  # 56x56 -> 28x28
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0)  # 28x28 -> 14x14
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112 -> 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x