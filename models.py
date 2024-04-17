import torch
import torch.nn as nn



class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.handle_input = None
        if in_channels != out_channels:
            self.handle_input = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.handle_input:
            x = self.handle_input(x)
        return nn.functional.gelu(out + x)
    

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x, x_skip):
        return self.model(torch.cat([x, x_skip], axis=1))
    

class UNet(nn.Module):
    def __init__(self, in_channels, height, width, hidden_channels, n_downs=4):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.hidden_channels = hidden_channels
        self.n_downs = n_downs

        # Define initial block
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1),
            nn.GELU()
        )

        # Define down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2**i*hidden_channels, 2**(i+1)*hidden_channels))

        # Define layers at the center
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height//2**n_downs, width//2**n_downs)),
            nn.GELU()
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2**n_downs*hidden_channels, 
                               2**n_downs*hidden_channels, 
                               (height//2**n_downs, width//2**n_downs)),
            nn.BatchNorm2d(2**n_downs*hidden_channels),
            nn.GELU()
        )
        
        # Define up blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downs-1, -1, -1):
            self.up_blocks.append(UNetUp(2**(i+2)*hidden_channels, 2**i*hidden_channels))
        
        # Define final block
        self.final_conv = nn.Sequential(
            nn.Conv2d(2*hidden_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, 1, 1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0: downs.append(down_block(x))
            else: downs.append(down_block(downs[-1]))
        
        up = self.up0(self.to_vec(downs[-1]))
        for up_block, down in zip(self.up_blocks, downs[::-1]):
            up = up_block(up, down)
        return self.final_conv(torch.cat([up, x], axis=1))


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_downs=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_downs = n_downs

        # Define initial block
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1),
            nn.GELU()
        )

        # Define down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(hidden_channels//2**i, hidden_channels//2**(i+1)))
        
        # define final block
        self.final_conv = nn.Conv2d(hidden_channels//2**n_downs, 1, 1, 1)

    def forward(self, x):
        x = self.init_conv(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        return self.final_conv(x)



if __name__ == "__main__":
    x = torch.randn(4, 3, 5, 5)
    in_channels = 3
    out_channels = 6
    res_block = ResidualConvBlock(in_channels, out_channels)
    print(x.shape)
    print(res_block(x).shape)

    x = torch.randn(3, 4, 32, 32)
    unet_down = UNetDown(x.shape[1], 10)
    print(x.shape)
    print(unet_down(x).shape)

    device = torch.device("mps")
    x = torch.randn(3, 3, 32, 32).to(device)
    unet = UNet(3, 32, 32, 64).to(device)
    print(x.shape)
    print(unet(x).shape)

    disc = Discriminator(3, 64).to(device)
    print(x.shape)
    print(disc(x).shape)

