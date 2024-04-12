import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models import UNet, Discriminator
from utils import Horse2zebraDataset
import os
from zipfile import ZipFile
from tqdm import tqdm



class CycleGAN(nn.Module):
    def __init__(self, checkpoint_name=None, dataset_name=None, device="cuda"):
        super(CycleGAN, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.file_dir = os.path.dirname(__file__)
        self.dataset_name = self.get_dataset_name(dataset_name, checkpoint_name, self.file_dir)
        self.device = torch.device(device)
        self.gen_AB = self.initialize_generator(self.dataset_name, checkpoint_name, self.device, self.file_dir, "gen_AB")
        self.gen_BA = self.initialize_generator(self.dataset_name, checkpoint_name, self.device, self.file_dir, "gen_BA")

    
    def train(self, n_epochs, batch_size, lr):
        dataloader = DataLoader(self.instantiate_dataset(self.dataset_name, self.get_transform(self.dataset_name), self.file_dir), 
                                batch_size, True)
        disc_A = self.initialize_discriminator(self.dataset_name, self.checkpoint_name, self.device, self.file_dir, "disc_A")
        disc_B = self.initialize_discriminator(self.dataset_name, self.checkpoint_name, self.device, self.file_dir, "disc_B")
        gen_optimizer = self.initialize_gen_optimizer(self.gen_AB, self.gen_BA, lr, self.checkpoint_name, self.file_dir, self.device)
        disc_optimizer = self.initialize_disc_optimizer(disc_A, disc_B, lr, self.checkpoint_name, self.file_dir, self.device)

        for epoch in range(n_epochs):
            for realA, realB in tqdm(dataloader):
                realA = realA.to(self.device)
                realB = realB.to(self.device)
                fakeB = self.gen_AB(realA)
                fakeA = self.gen_BA(realB)

                # update discriminator
                disc_optimizer.zero_grad()
                disc_loss = self.get_disc_loss(disc_A, disc_B, realA, realB, fakeA, fakeB)
                disc_loss.backward()
                disc_optimizer.step()

                # update generator
                gen_optimizer.zero_grad()
                gen_loss = self.get_gen_loss()





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
    
    def instantiate_dataset(self, dataset_name, transform, file_dir, train=True):
        """Instantiate dataset for given dataset name"""
        if dataset_name == "horse2zebra":
            return Horse2zebraDataset(os.path.join(file_dir, "datasets", dataset_name), transform, train)

    def unzip_dataset(self, dataset_name, file_dir):
        if dataset_name == "horse2zebra":
            extract_to = os.path.join(file_dir, "datasets", dataset_name)
            if os.path.exists(extract_to):
                print(f"Directory {extract_to} already exists. No operation done")
            else:
                os.mkdir(extract_to)
                for file_name in ["horse2zebraA.zip", "horse2zebraB.zip"]:
                    with ZipFile(os.path.join(file_dir, "datasets", file_name), "r") as zip_file:
                        zip_file.extractall(extract_to)

    def get_transform(self, dataset_name):
        """Returns the transform object for a given dataset name"""
        if dataset_name == "horse2zebra":
            return transforms.Compose([transforms.ToTensor(), lambda x: 2*x - 1])

    def initialize_disc_optimizer(self, disc_A, disc_B, lr, checkpoint_name, file_dir, device):
        """Initializes discriminator optimizer"""
        disc_optimizer = optim.Adam(list(disc_A.parameters()) + list(disc_B.parameters()), lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        return disc_optimizer
    
    def initialize_gen_optimizer(self, gen_AB, gen_BA, lr, checkpoint_name, file_dir, device):
        """Initializes generator optimizer"""
        gen_optimizer = optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters), lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint), device)
            gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        return gen_optimizer
                
    def get_disc_loss(self, disc_A, disc_B, realA, realB, fakeA, fakeB):
        """Returns discriminator loss"""
        disc_loss_A = nn.functional.l1_loss(disc_A(fakeA.detach()), disc_A(realA))
        disc_loss_B = nn.functional.l1_loss(disc_B(fakeB.detach()), disc_B(realB))
        return disc_loss_A + disc_loss_B


