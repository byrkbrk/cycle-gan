import torch
import torch.nn as nn
from models import UNet, Discriminator



class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
    
    def train(self, n_epochs, batch_size, lr, checkpoint_name):
        pass

    def generate(self, n_samples, checkpoint_name):
        pass
