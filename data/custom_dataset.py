import os 
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.img_dir = data_dir

        # Dynamically create the mapping for class labels
        self.class_mapping = {class_name: idx for idx, class_name in enumerate(self.data["Class"].unique())}
        self.reverse_class_mapping = {idx: class_name for class_name, idx in self.class_mapping.items()}
        
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["FileName"])
        image = Image.open(img_path).convert("RGB")

        # Map class labels to integers
        label = self.class_mapping[row["Class"]]
        more_than_one_person = 1 if row["MoreThanOnePerson"].strip().lower() == "yes" else 0

        if self.transform: 
            image = self.transform(image)
        labels = torch.tensor([label, more_than_one_person], dtype=torch.float)
        return image, labels
