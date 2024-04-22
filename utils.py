import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from zipfile import ZipFile
import requests
from torchvision.utils import save_image



class Horse2zebraDataset(Dataset):
    def __init__(self, dataset_dir, transform, train):
        if train:
            self.dataset_pathA = os.path.join(dataset_dir, "trainA")
            self.dataset_pathB = os.path.join(dataset_dir, "trainB")
        else:
            self.dataset_pathA = os.path.join(dataset_dir, "testA")
            self.dataset_pathB = os.path.join(dataset_dir, "testB")
        self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in os.listdir(self.dataset_pathA)]
        self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in os.listdir(self.dataset_pathB)]
        self.transform = transform

    def __getitem__(self, index):
        if index < len(self.image_pathsA): imageA = self.read_image(self.image_pathsA[index])
        else: imageA = self.read_image(self.image_pathsA[torch.randint(len(self.image_pathsA), (1, ))])

        if index < len(self.image_pathsB): imageB = self.read_image(self.image_pathsB[index])
        else: imageB = self.read_image(self.image_pathsB[torch.randint(len(self.image_pathsB), (1, ))])
        return self.transform(imageA), self.transform(imageB)

    def __len__(self):
        return max(len(self.image_pathsA), len(self.image_pathsB))
    
    def read_image(self, img_path):
        return Image.open(img_path)


def unzip_dataset(dataset_name, file_dir):
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


def download_dataset(dataset_name, root, base_folder, url=None):
    """Downloads dataset into root/base_folder directory"""
    if url is None:
        url = f"https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip"
    file_path = os.path.join(root, base_folder, f"{dataset_name}.zip")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. No operation done!")
        return
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File {file_path} downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)


if __name__ == "__main__":
    # unzip_dataset("horse2zebra", os.path.dirname(__file__))
    # dataset_dir = os.path.join(os.path.dirname(__file__), "datasets", "horse2zebra")
    # dataset = Horse2zebraDataset(dataset_dir, transforms.ToTensor(), True)

    # itr = iter(dataset)
    # shapes = set()
    # for i in range(3000):
    #     a, b = next(itr)
    #     print(i, a.shape, b.shape)
    #     shapes.add(a.shape)
    #     shapes.add(b.shape)
    #     if b.shape[0] == 1:
    #         print(b.shape)
    #         save_image(b, "single_channel.jpeg")
    #         c = b.repeat(3, 1, 1)
    #         save_image(c, "repeated_channels.jpeg")
    #         break
        

    # print(shapes)
    dataset_name = "facades"
    file_dir = os.path.dirname(__file__)
    download_dataset(dataset_name, file_dir)