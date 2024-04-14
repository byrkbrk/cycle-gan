import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
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
        self.create_dirs(self.file_dir)
    
    def train(self, n_epochs, batch_size, lr, criterion_name="L1", lambda_id=0.1, lambda_cycle=10):
        dataloader = self.instantiate_dataloader(batch_size, self.dataset_name, self.checkpoint_name, self.file_dir)
        disc_A = self.initialize_discriminator(self.dataset_name, self.checkpoint_name, self.device, self.file_dir, "disc_A")
        disc_B = self.initialize_discriminator(self.dataset_name, self.checkpoint_name, self.device, self.file_dir, "disc_B")
        gen_optimizer = self.initialize_gen_optimizer(self.gen_AB, self.gen_BA, lr, self.checkpoint_name, self.file_dir, self.device)
        disc_optimizer = self.initialize_disc_optimizer(disc_A, disc_B, lr, self.checkpoint_name, self.file_dir, self.device)
        criterion = self.instantiate_criterion(criterion_name)

        avg_disc_loss = avg_gen_loss = 0
        for epoch in range(n_epochs):
            for realA, realB in tqdm(dataloader, desc=f"Epoch {epoch}"):
                realA = realA.to(self.device)
                realB = realB.to(self.device)
                fakeB = self.gen_AB(realA)
                fakeA = self.gen_BA(realB)

                # update discriminator
                disc_optimizer.zero_grad()
                disc_loss = self.get_disc_loss(disc_A, disc_B, realA, realB, fakeA, fakeB, criterion)
                disc_loss.backward()
                disc_optimizer.step()

                # update generator
                gen_optimizer.zero_grad()
                gen_loss = self.get_gen_loss(self.gen_AB, self.gen_BA, disc_A, disc_B, realA, realB, fakeA, fakeB,
                                             criterion, criterion, criterion, lambda_id, lambda_cycle, )
                gen_loss.backward()
                gen_optimizer.step()

                avg_disc_loss += disc_loss.item()/len(dataloader)
                avg_gen_loss += gen_loss.item()/len(dataloader)
            print(f"Epoch: {epoch}, Discriminator loss: {avg_disc_loss}, Generator loss {avg_gen_loss}")
            self.save_tensor_images(realA, fakeA, realB, fakeB, epoch, self.file_dir)
            avg_disc_loss = avg_gen_loss = 0
            

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
            gen = UNet(3, 256, 256, 32, n_downs=3).to(device)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            gen.load_state_dict(checkpoint[gen_name])
        return gen
    
    def initialize_discriminator(self, dataset_name, checkpoint_name, device, file_dir, disc_name):
        """Returns initialized discriminator for given inputs"""
        if dataset_name == "horse2zebra":
            disc = Discriminator(3, 32, n_downs=3).to(device)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            disc.load_state_dict(checkpoint[disc_name])
        return disc
    
    def instantiate_dataset(self, dataset_name, transform, file_dir, train=True):
        """Instantiate dataset for given dataset name"""
        if dataset_name == "horse2zebra":
            self.unzip_dataset(dataset_name, file_dir)
            return Horse2zebraDataset(os.path.join(file_dir, "datasets", dataset_name), transform, train)

    def unzip_dataset(self, dataset_name, file_dir):
        """Unzip dataset for given dataset name"""
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
            return transforms.Compose([transforms.ToTensor(), 
                                       lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x, # handle 1-channel images
                                       lambda x: 2*x - 1]) # pixels to [-0.5, 0.5]

    def initialize_disc_optimizer(self, disc_A, disc_B, lr, checkpoint_name, file_dir, device):
        """Initializes discriminator optimizer"""
        disc_optimizer = optim.Adam(list(disc_A.parameters()) + list(disc_B.parameters()), lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        return disc_optimizer
    
    def initialize_gen_optimizer(self, gen_AB, gen_BA, lr, checkpoint_name, file_dir, device):
        """Initializes generator optimizer"""
        gen_optimizer = optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint), device)
            gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        return gen_optimizer
                
    def get_disc_loss(self, disc_A, disc_B, realA, realB, fakeA, fakeB, criterion):
        """Returns discriminator loss"""
        disc_loss_A = self.get_disc_X_loss(disc_A, realA, fakeA, criterion)
        disc_loss_B = self.get_disc_X_loss(disc_B, realB, fakeB, criterion)
        return disc_loss_A + disc_loss_B
    
    def get_disc_X_loss(self, disc_X, realX, fakeX, criterion):
        """Returns discriminator X loss"""
        pred_real = disc_X(realX)
        pred_fake = disc_X(fakeX.detach())
        return 1/2*(criterion(pred_real, torch.ones_like(pred_real)) + criterion(pred_fake, torch.zeros_like(pred_fake)))
    
    def get_gen_loss(self, gen_AB, gen_BA, disc_A, disc_B, realA, realB, fakeA, fakeB, 
                     id_criterion, cycle_criterion, adv_criterion, lambda_id, lambda_cycle):
        """Returns generator loss"""
        id_loss_AB = self.get_id_loss(gen_AB, realB, id_criterion, lambda_id)
        cycle_loss_A = self.get_cycle_loss(gen_BA, fakeB, realA, cycle_criterion, lambda_cycle)
        adv_loss_AB = self.get_adv_loss(fakeB, disc_B, adv_criterion)

        id_loss_BA = self.get_id_loss(gen_BA, realA, id_criterion, lambda_id)
        cycle_loss_B = self.get_cycle_loss(gen_AB, fakeA, realB, cycle_criterion, lambda_cycle)
        adv_loss_BA = self.get_adv_loss(fakeA, disc_A, adv_criterion)
        return id_loss_AB + cycle_loss_A + adv_loss_AB + id_loss_BA + cycle_loss_B + adv_loss_BA

    def get_id_loss(self, gen_XY, realY, criterion, lambda_id):
        """Returns identity loss"""
        return lambda_id*criterion(gen_XY(realY), realY)
    
    def get_cycle_loss(self, gen_YX, fakeY, realX, criterion, lambda_cycle):
        """Returns cycle loss"""
        return lambda_cycle*criterion(gen_YX(fakeY), realX)
    
    def get_adv_loss(self, fakeY, disc_Y, criterion):
        """Returns adversarial loss"""
        pred = disc_Y(fakeY)
        return criterion(pred, torch.ones_like(pred))
    
    def instantiate_criterion(self, criterion_name="L1"):
        """Returns instantiated criterion"""
        assert criterion_name in {"L1", "mse"}, "Unknown criterion name"

        if criterion_name == "L1":
            return nn.L1Loss()
        elif criterion_name == "mse":
            return nn.MSELoss()
    
    def save_tensor_images(self, realA, fakeA, realB, fakeB, epoch, file_dir):
        """Save given tensor images into saved-images directory"""
        save_image(torch.cat([realA, fakeB, realB, fakeA], axis=0), 
                   os.path.join(file_dir, "saved-images", f"realA_fakeB_realB_fakeA_{epoch}.jpeg"), 
                   nrow=len(realA))

    def create_dirs(self, file_dir):
        """Create directories used in training and inferencing"""
        dir_names = ["checkpoints", "saved-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    def instantiate_dataloader(self, batch_size, dataset_name, checkpoint_name, file_dir):
        """Returns dataloader for given dataset name"""
        if checkpoint_name:
            batch_size = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name),
                                    torch.device("cpu"))["batch_size"]
        dataset = self.instantiate_dataset(dataset_name, self.get_transform(dataset_name), file_dir)
        return DataLoader(dataset, batch_size, True, drop_last=True)
    
    def save_checkpoint(self):
        pass
            




if __name__ == "__main__":
    cycle_gan = CycleGAN(None, "horse2zebra", "mps")
    cycle_gan.train(5, 4, 2e-4)