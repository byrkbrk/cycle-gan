import torch
import torch.nn as nn
from models import UNet, Discriminator
import os



class CycleGAN(nn.Module):
    def __init__(self, checkpoint_name=None, dataset_name=None, device="cuda"):
        super(CycleGAN, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.file_dir = os.path.dirname(__file__)
        self.dataset_name = self.get_dataset_name(dataset_name, checkpoint_name, self.file_dir)
        self.device = torch.device(device)
        self.gen_AB = self.initialize_generator(self.dataset_name, checkpoint_name, self.device, self.file_dir, "gen_AB")
        self.gen_BA = self.initialize_generator(self.dataset_name, checkpoint_name, self.device, self.file_dir, "gen_BA")

    
    def train(self, n_epochs, batch_size, lr, checkpoint_name):
        pass

    def generate(self, n_samples, checkpoint_name):
        pass

    def get_dataset_name(self, dataset_name, checkpoint_name, file_dir):
        """Returns dataset name based on if checkpoint-name provided"""
        if checkpoint_name:
            dataset_name = torch.load(
                os.path.join(file_dir, "checkpoints", checkpoint_name), 
                map_location=torch.device("cpu"))["dataset_name"]
        assert dataset_name in {"horse2zebra"}, "Unknown dataset name"
        return dataset_name
    
    def initialize_generator(self, dataset_name, checkpoint_name, device, file_dir, gen_name):
        """Returns initialized generator for given inputs"""
        if dataset_name == "horse2zebra":
            gen = UNet(3, 256, 256, 64).to(device)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            gen.load_state_dict(checkpoint[gen_name])
        return gen
    
    def initialize_discriminator(self, dataset_name, checkpoint_name, device, file_dir, disc_name):
        """Returns initialized discriminator for given inputs"""
        if dataset_name == "horse2zebra":
            disc = Discriminator(3, 64).to(device)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            disc.load_state_dict(checkpoint[disc_name])
        return disc
    
    def instantiate_dataset(self, dataset_name, transforms, file_dir, train=True):
        transform, target_transform = transforms
        if dataset_name == "horse2zebra":
            pass

        

