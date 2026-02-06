import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from .custom_dataset import CustomDataset


class TransformSubset:
    """
    Wrapper that applies a specific transform to a subset of data.
    Used to apply training augmentation only to the training split.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        # Get the underlying dataset index
        data_idx = self.subset.indices[idx]
        
        # Get image and labels from the underlying dataset WITHOUT transform
        dataset = self.subset.dataset
        row = dataset.data.iloc[data_idx]
        
        import os
        img_path = os.path.join(dataset.img_dir, row["FileName"])
        
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        
        # Apply THIS subset's transform
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        label = dataset.class_mapping[row["Class"]]
        more_than_one_person = 1 if row["MoreThanOnePerson"].strip().lower() == "yes" else 0
        labels = torch.tensor([label, more_than_one_person], dtype=torch.float)
        
        return image, labels
    
    def __len__(self):
        return len(self.subset)

class DataHandler:
    def __init__(self, csv_file, img_dir, augmentations, split_ratio=(0.7, 0.15, 0.15)):
        
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.augmentations = augmentations
        self.split_ratio = split_ratio

        # Create separate transforms for train and eval
        self.train_transform = self._create_train_transforms()
        self.eval_transform = self._create_eval_transforms()

        # Load the base dataset WITHOUT transforms initially
        base_dataset = CustomDataset(csv_file=self.csv_file, data_dir=self.img_dir, transform=None)
        
        # Store reference to base dataset for accessing class mappings
        self.dataset = base_dataset

        # Split the dataset
        train_subset, val_subset, test_subset = self._split_dataset(base_dataset)
        
        # Wrap subsets with appropriate transforms
        self.train_dataset = TransformSubset(train_subset, self.train_transform)
        self.val_dataset = TransformSubset(val_subset, self.eval_transform)
        self.test_dataset = TransformSubset(test_subset, self.eval_transform)

    def _create_train_transforms(self):
        """
        Create training transformations WITH data augmentation.
        """
        transforms_list = [
            transforms.Resize((224, 224)),
        ]
        
        # Add augmentation transforms (only for training)
        if self.augmentations.get("horizontal_flip", False):
            transforms_list.append(transforms.RandomHorizontalFlip())
        if self.augmentations.get("rotation", 0) > 0:
            transforms_list.append(transforms.RandomRotation(self.augmentations["rotation"]))
        
        # Add normalization at the end
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.augmentations["normaliszation"]["mean"],
                std=self.augmentations["normaliszation"]["std"]
            )
        ])

        return transforms.Compose(transforms_list)
    
    def _create_eval_transforms(self):
        """
        Create evaluation transformations WITHOUT data augmentation.
        Used for validation and test sets to ensure reproducible metrics.
        """
        transforms_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.augmentations["normaliszation"]["mean"],
                std=self.augmentations["normaliszation"]["std"]
            )
        ]

        return transforms.Compose(transforms_list)

    def _split_dataset(self, dataset):
        """
        Split the dataset into training, validation, and test subsets.
        """
        total_size = len(dataset)
        train_size = int(self.split_ratio[0] * total_size)
        val_size = int(self.split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size
        return random_split(dataset, [train_size, val_size, test_size])

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