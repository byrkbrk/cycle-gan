import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os



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



if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets", "horse2zebra")
    dataset = Horse2zebraDataset(dataset_dir, transforms.ToTensor(), True)

    iter = iter(dataset)
    for i in range(10):
        a, b = next(iter)
        print(i, a.shape, b.shape)


