import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from .custom_dataset import CustomDataset

class DataHandler:
    def __init__(self, csv_file, img_dir, augmentations, split_ratio=(0.7, 0.15, 0.15)):
        
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.augmentations = augmentations
        self.split_ratio = split_ratio

        # Initialize transforms
        self.transforms = self._create_transforms()

        # Load the dataset
        self.dataset = CustomDataset(csv_file=self.csv_file, data_dir=self.img_dir, transform=self.transforms)

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset()

    def _create_transforms(self):
        """
        Create a list of transformations based on the provided augmentations.
        """
        transforms_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.augmentations["normaliszation"]["mean"],
                std=self.augmentations["normaliszation"]["std"]
            )
        ]

        if self.augmentations.get("horizontal_flip", False):
            transforms_list.append(transforms.RandomHorizontalFlip())
        if self.augmentations.get("rotation", 0) > 0:
            transforms_list.append(transforms.RandomRotation(self.augmentations["rotation"]))

        return transforms.Compose(transforms_list)

    def _split_dataset(self):
        """
        Split the dataset into training and testing subsets.
        """
        total_size = len(self.dataset)
        train_size = int(self.split_ratio[0] * total_size)
        val_size = int(self.split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size
        return random_split(self.dataset, [train_size, val_size,test_size])

    def get_dataloaders(self, batch_size, num_workers):
        """
        Create DataLoaders for the training and testing datasets.
        """
        dataloaders = {
            "train": DataLoader(
                dataset=self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            "val": DataLoader(
                dataset=self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            "test": DataLoader(
                dataset=self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        }
        print(f"Training set size: {len(self.train_dataset)}, Validating set size: {len(self.val_dataset)}, Testing set size: {len(self.test_dataset)}")

        return dataloaders