import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)])
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.head(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_dim_mlp = 2048, output_dim = 10):
        super().__init__()
        hidden_dim = 64
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, stride = 2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride = 2)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride = 1)
        self.conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 3, stride = 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        return self.mlp(x.flatten(1))
    
class FullyConvolutionalNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        hidden_dim = 64
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride = 1,  padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride = 1,  padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.head = nn.Conv2d(hidden_dim*4, output_channels, 1,  stride  = 1, padding=0)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.head(x)
        return x
    
class NonSizeDependentCNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        hidden_dim = 64
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride = 1,  padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride = 1,  padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(hidden_dim*4, output_dim)
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
