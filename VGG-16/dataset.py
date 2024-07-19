import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import CustomTransform

def load_dataset(data_path, mean_r, mean_g, mean_b, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        CustomTransform(mean_r, mean_g, mean_b),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])  # Rescale to [0,1]
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader