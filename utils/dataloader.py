import os
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def get_dataloader(data_path, batch_size, resolution):
    dataset = CustomDataset(data_path, resolution)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


class CustomDataset(Dataset):
    def __init__(self, data_path, resolution):
        self.data_path = data_path
        self.image_files = os.listdir(data_path)

        self.transforms = Compose([
            Resize((resolution, resolution)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        return image
