import torch
import torch.nn as nn
import torch.nn.functional as F


# Generator model
class Generator(nn.Module):
    def __init__(self, vector_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(50, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, vector_shape),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.model(x)
    
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, vector_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(vector_shape, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)