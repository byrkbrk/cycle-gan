import torch
import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        layer = lambda in_channels, out_channels: [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()]
        self.model = nn.Sequential(*(layer(in_channels, out_channels) + layer(in_channels, out_channels)))

    def forward(self, x):
        return self.model(x) + x
    

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.model(x)
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_downs=2, n_residuals=9):
        super(Generator, self).__init__()
        layers = []
        
        # Define initial conv layer
        layers.append(nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 7, 1, 3, padding_mode="reflect"), nn.GELU()))

        # Define down blocks
        for i in range(n_downs):
            layers.append(DownBlock(2**i*hidden_channels, 2**(i+1)*hidden_channels))
        
        # Define residual blocks
        for _ in range(n_residuals):
            layers.append(ResidualBlock(2**n_downs*hidden_channels, 2**n_downs*hidden_channels))
        
        # Define up blocks
        for i in range(n_downs, 0, -1):
            layers.append(UpBlock(2**i*hidden_channels, 2**(i-1)*hidden_channels))
        
        # Define final conv layer
        layers.append(nn.Sequential(nn.Conv2d(hidden_channels, in_channels, 7, 1, 3, padding_mode="reflect"), nn.Tanh()))

        # Sequentialize model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        

class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_downs=3):
        super(Discriminator, self).__init__()
        layers = []

        # Define initial conv block
        layers.append(nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1), nn.GELU()))

        # Define down blocks
        for i in range(n_downs):
            layers.append(DownBlock(2**i*hidden_channels, 2**(i+1)*hidden_channels))
        
        # Define final conv block
        layers.append(nn.Conv2d(2**n_downs*hidden_channels, 1, 1))

        # Sequentialize model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    x = torch.randn(2, 8, 16, 16)
    res_block = ResidualBlock(8, 8)
    print(x.shape)
    print(res_block(x).shape)

    down_block = DownBlock(8, 16)
    print(x.shape)
    print(down_block(x).shape)

    x = torch.randn(4, 2, 16, 16)
    up_block = UpBlock(2, 1)
    print(x.shape)
    print(up_block(x).shape)

    device = torch.device("mps")
    x = torch.randn(4, 3, 16, 16).to(device)
    gen = Generator(3, 16).to(device)
    print(x.shape)
    print(gen(x).shape)

    print("Disc")
    disc = Discriminator(3, 16).to(device)
    print(x.shape)
    print(disc(x).shape)

        