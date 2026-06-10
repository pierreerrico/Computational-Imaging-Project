import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class ImageDataset(Dataset):
    def __init__(self, hf_dataset, image_size=256):
        self.hf_dataset = hf_dataset

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        image = self.hf_dataset[index]["image"]

        image = image.convert("RGB")

        return self.preprocess(image)