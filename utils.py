import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from zipfile import ZipFile
import requests
from torchvision.utils import save_image
import gdown



class Horse2zebraDataset(Dataset):
    """Horse2zebra dataset"""
    base_folder = dataset_name = "horse2zebra"
    def __init__(self, root, transform, train):
        unzip_dataset(self.dataset_name, self.base_folder, root)
        if train:
            self.dataset_pathA = os.path.join(root, self.base_folder, "trainA")
            self.dataset_pathB = os.path.join(root, self.base_folder, "trainB")
            self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in os.listdir(self.dataset_pathA)]
            self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in os.listdir(self.dataset_pathB)]
        else:
            self.dataset_pathA = os.path.join(root, self.base_folder, "testA")
            self.dataset_pathB = os.path.join(root, self.base_folder, "testB")
            self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in self.sort_files(os.listdir(self.dataset_pathA))]
            self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in self.sort_files(os.listdir(self.dataset_pathB))]
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
    
    def sort_files(self, files):
        """Sorts based on file indices for a given list of files"""
        f = lambda x: int(os.path.splitext(x)[0].split("_")[-1])
        return sorted(files, key=f)


class Monet2photoDataset(Dataset):
    """Monet2photo dataset"""
    base_folder = dataset_name = "monet2photo"
    def __init__(self, root, transform, train, download):
        if download:
            download_dataset(self.dataset_name, root)
        self.unzip_dataset(self.base_folder, root)
        
        if train:
            self.dataset_pathA = os.path.join(root, self.base_folder, "trainA")
            self.dataset_pathB = os.path.join(root, self.base_folder, "trainB")
            self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in os.listdir(self.dataset_pathA)]
            self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in os.listdir(self.dataset_pathB)]
        else:
            self.dataset_pathA = os.path.join(root, self.base_folder, "testA")
            self.dataset_pathB = os.path.join(root, self.base_folder, "testB")
            self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in self.sort_files(os.listdir(self.dataset_pathA))]
            self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in self.sort_files(os.listdir(self.dataset_pathB))]
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
    
    def sort_files(self, files):
        """Sorts based on file indices for a given list of files"""
        return sorted(files)
    
    def unzip_dataset(self, base_folder, root):
        """Unzip monet2photo.zip dataset"""
        if os.path.exists(os.path.join(root, base_folder)):
            print(f"Directory {os.path.join(root, base_folder)} already exists. No operation done")
            return
        file_names = ["monet2photo.zip"]
        extract_to = os.path.join(root, "")
        for file_name in file_names:
            with ZipFile(os.path.join(root, file_name), "r") as zip_file:
                    zip_file.extractall(extract_to)
        

class Latex2handwrittenDataset(Dataset):
    """Latex2handwritten dataset"""
    base_folder = dataset_name = "latex2handwritten"
    def __init__(self, root, transform, train):
        self.download_dataset(__class__.dataset_name, root)
        unzip_dataset(__class__.dataset_name, __class__.base_folder, root)
        if train:
            self.dataset_pathA = os.path.join(root, self.base_folder, "trainA")
            self.dataset_pathB = os.path.join(root, self.base_folder, "trainB")
            self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in os.listdir(self.dataset_pathA)]
            self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in os.listdir(self.dataset_pathB)]
        else:
            self.dataset_pathA = os.path.join(root, self.base_folder, "testA")
            self.dataset_pathB = os.path.join(root, self.base_folder, "testB")
            self.image_pathsA = [os.path.join(self.dataset_pathA, file) for file in self.sort_files(os.listdir(self.dataset_pathA))]
            self.image_pathsB = [os.path.join(self.dataset_pathB, file) for file in self.sort_files(os.listdir(self.dataset_pathB))]
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
    
    def sort_files(self, files):
        """Sorts based on file indices for a given list of files"""
        #f = lambda x: int(os.path.splitext(x)[0].split("_")[-1])
        return sorted(files)
    
    def download_dataset(self, dataset_name, root):
        """Downloads dataset for given dataset name"""
        file_name_id = {"latex2handwritten": ["latex2handwritten.zip", 
                                        "1yx9cCjdKTednizft1piqMchsYZW4E5yp"]}
        file_name, file_id = file_name_id[dataset_name]
        if os.path.exists(os.path.join(root, file_name)):
            print(f"File exists! No operation done: {os.path.join(root, file_name)}")
            return
        else:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output=os.path.join(root, file_name), quiet=False)


def unzip_dataset(dataset_name, base_folder, root):
        """Unzip dataset for given dataset name"""
        if dataset_name == "horse2zebra":
            file_names =  ["horse2zebraA.zip", "horse2zebraB.zip"]
        elif dataset_name == "monet2photo":
            file_names = ["monet2photo.zip"]
        elif dataset_name == "latex2handwritten":
            file_names = ["latex2handwritten.zip"]
        else:
            raise ValueError(f"Undefined dataset name: {dataset_name}")
        
        extract_to = os.path.join(root, base_folder)
        print(dataset_name, extract_to)
        if os.path.exists(extract_to):
                print(f"Directory {extract_to} already exists. No operation done")
                return
        os.mkdir(extract_to)
        
        for file_name in file_names:
                with ZipFile(os.path.join(root, file_name), "r") as zip_file:
                    zip_file.extractall(extract_to)


def download_dataset(dataset_name, root, url=None):
    """Downloads dataset into root/ directory"""
    if url is None:
        url = f"https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip"
    file_path = os.path.join(root, f"{dataset_name}.zip")
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. No operation done!")
        return
    
    print(f"Sending request to url {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File {file_path} downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)


class ImageBuffer(object):
    """Keeps images in a specified-size buffer"""
    def __init__(self, buffer_capacity=None):
        self.buffer = []
        self.buffer_capacity = buffer_capacity
    
    def get_tensor(self, images):
        """Returns images from buffer"""
        return torch.cat([self._push_and_pop(image.detach()[None]) for image in images])

    def _push_and_pop(self, image):
        """Pushes (if available) into buffer and returns (if possible) given image"""
        if self.buffer_capacity == 0: return image

        if self.size() < self.buffer_capacity:
            self.buffer.append(image)
        else:
            if torch.rand(1) > 0.5:
                idx = torch.randint(self.buffer_capacity, (1, ))
                image, self.buffer[idx] = self.buffer[idx], image
        return image

    def size(self):
        """Returns the size of the buffer"""
        return len(self.buffer)
    
    def state_dict(self):
        """Returns state dictionary of image-buffer class"""
        return {"buffer": self.buffer, "buffer_capacity": self.buffer_capacity}
    
    def load_state_dict(self, state_dict):
        """Loads given buffer state dictionary"""
        self.buffer = state_dict["buffer"]
        self.buffer_capacity = state_dict["buffer_capacity"]

def download_checkpoint(dataset_name, root, base_folder="checkpoints"):
    """Downloads pretrained checkpoint for given dataset name"""
    file_name_id = {"horse2zebra": ["pretrained_horse2zebra_checkpoint_219.pth", 
                                    "10ZlokluOgzIfaeSN_CDJ277fJYAoIOXj"]}
    file_name, file_id = file_name_id[dataset_name]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=os.path.join(root, base_folder, file_name), quiet=False)
