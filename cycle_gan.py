import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from alternative_models import Generator, Discriminator
from utils import Horse2zebraDataset, ImageBuffer
import os
from zipfile import ZipFile
from tqdm import tqdm
import matplotlib.pyplot as plt



class CycleGAN(nn.Module):
    def __init__(self, checkpoint_name=None, dataset_name=None, device=None):
        super(CycleGAN, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.file_dir = os.path.dirname(__file__)
        self.dataset_name = self.get_dataset_name(dataset_name, checkpoint_name, self.file_dir)
        self.device = self.initialize_device(device)
        self.gen_AB = self.initialize_generator(self.dataset_name, checkpoint_name, self.device, self.file_dir, "gen_AB")
        self.gen_BA = self.initialize_generator(self.dataset_name, checkpoint_name, self.device, self.file_dir, "gen_BA")
        self.create_dirs(self.file_dir)
    
    def train(self, n_epochs, batch_size, lr, id_criterion_name="L1", cycle_criterion_name="L1", adv_criterion_name="mse", lambda_id=0.1, lambda_cycle=10, 
              checkpoint_save_dir=None, checkpoint_save_freq=1, image_save_dir=None, buffer_capacity=50):
        dataloader = self.instantiate_dataloader(batch_size, self.dataset_name, self.checkpoint_name, self.file_dir, use_train_set=True)
        disc_A = self.initialize_discriminator(self.dataset_name, self.checkpoint_name, self.device, self.file_dir, "disc_A")
        disc_B = self.initialize_discriminator(self.dataset_name, self.checkpoint_name, self.device, self.file_dir, "disc_B")
        gen_optimizer = self.initialize_gen_optimizer(self.gen_AB, self.gen_BA, lr, self.checkpoint_name, self.file_dir, self.device)
        disc_optimizer = self.initialize_disc_optimizer(disc_A, disc_B, lr, self.checkpoint_name, self.file_dir, self.device)
        gen_scheduler = self.initialize_scheduler(gen_optimizer, self.checkpoint_name, self.file_dir, self.device, "gen")
        disc_scheduler = self.initialize_scheduler(disc_optimizer, self.checkpoint_name, self.file_dir, self.device, "disc")
        id_criterion = self.instantiate_criterion(id_criterion_name)
        cycle_criterion = self.instantiate_criterion(cycle_criterion_name)
        adv_criterion = self.instantiate_criterion(adv_criterion_name)
        loss_dict = self._initialize_loss_dict(self.checkpoint_name, self.file_dir)
        buffer_fakeA = self.initialize_image_buffer(buffer_capacity, self.device, self.checkpoint_name, self.file_dir, "fakeA")
        buffer_fakeB = self.initialize_image_buffer(buffer_capacity, self.device, self.checkpoint_name, self.file_dir, "fakeB")

        for epoch in range(self.get_start_epoch(self.checkpoint_name, self.file_dir), 
                           self.get_start_epoch(self.checkpoint_name, self.file_dir) + n_epochs):
            for realA, realB in tqdm(dataloader, desc=f"Epoch {epoch}"):
                realA = realA.to(self.device)
                realB = realB.to(self.device)
                fakeB = self.gen_AB(realA)
                fakeA = self.gen_BA(realB)

                # update discriminator
                disc_optimizer.zero_grad()
                disc_loss = self.get_disc_loss(disc_A, disc_B, realA, realB, 
                                               buffer_fakeA.get_tensor(fakeA), buffer_fakeB.get_tensor(fakeB), adv_criterion, loss_dict)
                disc_loss.backward()
                disc_optimizer.step()

                # update generator
                gen_optimizer.zero_grad()
                gen_loss = self.get_gen_loss(self.gen_AB, self.gen_BA, disc_A, disc_B, realA, realB, fakeA, fakeB,
                                            id_criterion, cycle_criterion, adv_criterion, lambda_id, lambda_cycle, loss_dict)
                gen_loss.backward()
                gen_optimizer.step()
            gen_scheduler.step()
            disc_scheduler.step()
            self._average_temp_loss(loss_dict)
            print(f"Epoch: {epoch}, Discriminator loss: {loss_dict['Discriminator'][-1]}, Generator loss: {loss_dict['Generator'][-1]}")
            self.save_tensor_images(realA, fakeA, realB, fakeB, epoch, self.file_dir, image_save_dir)
            self._save_loss_figure(loss_dict, self.dataset_name, self.file_dir)
            if (epoch + 1) % checkpoint_save_freq == 0:
                self.save_checkpoint(self.gen_AB, self.gen_BA, gen_optimizer, disc_A, disc_B, disc_optimizer,
                                     gen_scheduler, disc_scheduler, buffer_fakeA, buffer_fakeB, epoch, batch_size, 
                                     self.dataset_name, loss_dict, self.device, self.file_dir, checkpoint_save_dir)


    def generate(self, gen_name="AB", use_train_set=False):
        """Generates images for given generator name, using test (or train) set"""
        save_dir = os.path.join(self.file_dir, "generated-images", f"{self.dataset_name}-{gen_name}")
        os.makedirs(save_dir, exist_ok=True)

        if gen_name == "AB": gen = self.gen_AB
        else: gen = self.gen_BA
        gen.eval()
        inference_transform = lambda x: (x + 1)/2
        for i, (realA, realB) in enumerate(self.instantiate_dataloader(1, self.dataset_name, None, self.file_dir, use_train_set, False, False)):
            if gen_name == "AB": image = realA.to(self.device)
            else: image = realB.to(self.device)
            save_image(torch.cat([inference_transform(image), inference_transform(gen(image))]), 
                       os.path.join(save_dir, f"image_{gen_name}_{i}.jpeg"))

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
            gen = Generator(3, 64).apply(self._initialize_weights).to(device)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            gen.load_state_dict(checkpoint[f"{gen_name}_state_dict"])
        return gen
    
    def initialize_discriminator(self, dataset_name, checkpoint_name, device, file_dir, disc_name):
        """Returns initialized discriminator for given inputs"""
        if dataset_name == "horse2zebra":
            disc = Discriminator(3, 64).apply(self._initialize_weights).to(device)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            disc.load_state_dict(checkpoint[f"{disc_name}_state_dict"])
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
                                       transforms.RandomHorizontalFlip(),
                                       lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x, # handle 1-channel images
                                       lambda x: 2*x - 1]) # pixels to [-1, 1]

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
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), device)
            gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        return gen_optimizer
                
    def get_disc_loss(self, disc_A, disc_B, realA, realB, fakeA, fakeB, criterion, loss_dict):
        """Returns discriminator loss"""
        disc_loss_A = self.get_disc_X_loss(disc_A, realA, fakeA, criterion)
        disc_loss_B = self.get_disc_X_loss(disc_B, realB, fakeB, criterion)
        disc_loss = disc_loss_A + disc_loss_B
        for key, loss_val in zip(["Discriminator-A", "Discriminator-B", "Discriminator"], 
                                 [disc_loss_A.item(), disc_loss_B.item(), disc_loss.item()]):
            loss_dict["temp-" + key].append(loss_val)
        return disc_loss
    
    def get_disc_X_loss(self, disc_X, realX, fakeX, criterion):
        """Returns discriminator X loss"""
        pred_real = disc_X(realX)
        pred_fake = disc_X(fakeX.detach())
        return 1/2*(criterion(pred_real, torch.ones_like(pred_real)) + criterion(pred_fake, torch.zeros_like(pred_fake)))
    
    def get_gen_loss(self, gen_AB, gen_BA, disc_A, disc_B, realA, realB, fakeA, fakeB, 
                     id_criterion, cycle_criterion, adv_criterion, lambda_id, lambda_cycle, loss_dict):
        """Returns generator loss"""
        id_loss_AB = self.get_id_loss(gen_AB, realB, id_criterion, lambda_id)
        cycle_loss_A = self.get_cycle_loss(gen_BA, fakeB, realA, cycle_criterion, lambda_cycle)
        adv_loss_AB = self.get_adv_loss(fakeB, disc_B, adv_criterion)
        id_loss_BA = self.get_id_loss(gen_BA, realA, id_criterion, lambda_id)
        cycle_loss_B = self.get_cycle_loss(gen_AB, fakeA, realB, cycle_criterion, lambda_cycle)
        adv_loss_BA = self.get_adv_loss(fakeA, disc_A, adv_criterion)
        gen_loss = id_loss_AB + cycle_loss_A + adv_loss_AB + id_loss_BA + cycle_loss_B + adv_loss_BA
        
        for key, loss_val in zip(["GenAB-identity", "Cycle-consistency-A", "GenAB-adversarial", 
                              "GenBA-identity", "Cycle-consistency-B", "GenBA-adversarial", "Generator"],
                              [id_loss_AB.item(), cycle_loss_A.item(), adv_loss_AB.item(),
                               id_loss_BA.item(), cycle_loss_B.item(), adv_loss_BA.item(), gen_loss.item()]):
            loss_dict["temp-" + key].append(loss_val)
        return gen_loss

    def get_id_loss(self, gen_XY, realY, criterion, lambda_id):
        """Returns identity loss"""
        return lambda_id*criterion(gen_XY(realY), realY)
    
    def get_cycle_loss(self, gen_YX, fakeY, realX, criterion, lambda_cycle):
        """Returns cycle consistency loss"""
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
    
    def save_tensor_images(self, realA, fakeA, realB, fakeB, epoch, file_dir, image_save_dir, inference_transform=lambda x: (x+1)/2):
        """Save given tensor images into saved-images directory"""
        if image_save_dir:
            fp = os.path.join(image_save_dir, f"realA_fakeB_realB_fakeA_{epoch}.jpeg")
        else:
            fp = os.path.join(file_dir, "saved-images", f"realA_fakeB_realB_fakeA_{epoch}.jpeg")
        save_image(torch.cat([inference_transform(image) for image in [realA, fakeB, realB, fakeA]]), fp, nrow=len(realA))

    def create_dirs(self, file_dir):
        """Create directories used in training and inferencing"""
        dir_names = ["checkpoints", "saved-images", "loss-figures"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    def instantiate_dataloader(self, batch_size, dataset_name, checkpoint_name, file_dir, use_train_set=True, shuffle=True, drop_last=True):
        """Returns dataloader for given dataset name"""
        dataset = self.instantiate_dataset(dataset_name, self.get_transform(dataset_name), file_dir, use_train_set)

        if checkpoint_name:
            batch_size = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name),
                                    torch.device("cpu"))["batch_size"]
        return DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    
    def save_checkpoint(self, gen_AB, gen_BA, gen_optimizer, disc_A, disc_B, disc_optimizer,
                        gen_scheduler, disc_scheduler, buffer_fakeA, buffer_fakeB, epoch, batch_size, dataset_name, 
                        loss_dict, device, file_dir, save_dir=None):
        """Saves checkpoint for given variables"""
        checkpoint = {
            "gen_AB_state_dict": gen_AB.state_dict(),
            "gen_BA_state_dict": gen_BA.state_dict(),
            "gen_optimizer_state_dict": gen_optimizer.state_dict(),
            "disc_A_state_dict": disc_A.state_dict(),
            "disc_B_state_dict": disc_B.state_dict(),
            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
            "gen_scheduler_state_dict": gen_scheduler.state_dict(),
            "disc_scheduler_state_dict": disc_scheduler.state_dict(),
            "buffer_fakeA_state_dict": buffer_fakeA.state_dict(),
            "buffer_fakeB_state_dict": buffer_fakeB.state_dict(),
            "epoch": epoch,
            "batch_size": batch_size,
            "dataset_name": dataset_name,
            "loss_dict": loss_dict,
            "device": device
        }

        if save_dir: 
            fpath = os.path.join(save_dir, f"{dataset_name}_checkpoint_{epoch}.pth")
        else: 
            fpath = os.path.join(file_dir, "checkpoints", f"{dataset_name}_checkpoint_{epoch}.pth")
        torch.save(checkpoint, fpath)            
    
    def initialize_device(self, device):
        """Initializes device based on device-availability if device info not provided"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def get_start_epoch(self, checkpoint_name, file_dir):
        """Returns starting epoch for training"""
        start_epoch = 0
        if checkpoint_name:
            start_epoch = torch.load(
                os.path.join(file_dir, "checkpoints", checkpoint_name), 
                map_location=torch.device("cpu"))["epoch"] + 1
        return start_epoch
    
    def _initialize_loss_dict(self, checkpoint_name, file_dir):
        """Returns loss dictionary"""
        if checkpoint_name:
            try:
                loss_dict = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), torch.device("cpu"))["loss_dict"]
                return loss_dict
            except KeyError:
                pass
        keys = ["GenAB-adversarial", "Cycle-consistency-A", "GenAB-identity",
                "GenBA-adversarial", "Cycle-consistency-B", "GenBA-identity",
                "Discriminator-A", "Discriminator-B",
                "Generator", "Discriminator"]
        loss_dict = {key: [] for key in keys}
        for key in list(loss_dict.keys()): loss_dict["temp-" + key] = []
        return loss_dict
    
    def _average_temp_loss(self, loss_dict):
        """Averages temporary losses and append relavent loss in loss dict"""
        pat = "temp-"
        for loss_name in loss_dict.keys():
            if pat in loss_name:
                loss_dict[loss_name[len(pat):]].append(sum(loss_dict[loss_name]) / len(loss_dict[loss_name]))
                loss_dict[loss_name] = []
    
    def _save_loss_figure(self, loss_dict, dataset_name, file_dir):
        """Creates and saves loss figure for given loss dict"""
        line_types = ['b-', 'g--', 'r:', 'c-.', 'm-', 'y--', 'k:', 'm-.', 'bx-', 'ro-.']
        labels = [label for label in loss_dict.keys() if "temp-" not in label]
        fig, ax = plt.subplots()
        for label, line_type in zip(labels, line_types):
            ax.plot(range(len(loss_dict[label])), loss_dict[label], line_type, label=label)
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        fig.savefig(os.path.join(file_dir, "loss-figures", f"{dataset_name}_loss_fig.png"))
        plt.close(fig)

    def initialize_scheduler(self, optimizer, checkpoint_name, file_dir, device, choice="gen", start_epoch=100, end_epoch=200):
        """Returns scheduler (of either generator or discriminator) for given choice"""
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._get_lr_lambda(start_epoch, end_epoch))
        if checkpoint_name:
            try:
                checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), device)
                scheduler.load_state_dict(checkpoint[choice + "_scheduler_state_dict"])
            except KeyError:
                # Handle unavailable scheduler checkpoint
                scheduler.last_epoch = self.get_start_epoch(checkpoint_name, file_dir)
        return scheduler

    def _get_lr_lambda(self, start_epoch=100, end_epoch=200):
        """Returns lr_lambda for LambdaLr scheduler"""
        return lambda epoch, s_epoch=start_epoch, e_epoch=end_epoch: (e_epoch-epoch)/(e_epoch - s_epoch) if epoch > s_epoch else 1

    def _initialize_weights(self, m, mean=0, std=0.02):
        """Initializes weights of model m with normal distribution"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean, std)

    def initialize_image_buffer(self, buffer_capacity, device, checkpoint_name, file_dir, choice="fakeA"):
        """Returns an initialized instance of ImageBuffer"""
        image_buffer = ImageBuffer(buffer_capacity)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), device)
            image_buffer.load_state_dict(checkpoint[f"buffer_{choice}_state_dict"])
        return image_buffer

if __name__ == "__main__":
    checkpoint_name = "horse2zebra_checkpoint_0.pth"
    cycle_gan = CycleGAN(checkpoint_name, "horse2zebra")
    cycle_gan.train(5, 2, 2e-4)