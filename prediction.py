import os
import pandas as pd
import random
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from utils.config_util import load_ymal_config
from data.data import DataHandler

# Import your trained model
from models.baseline import MultiTaskModel  # Replace with the actual filename containing your model
from models.multitask import AttentionModel

def load_model(model_path, device, config):
    # Initialize model
    if config["model"]["type"] == "baseline":
        model = MultiTaskModel(num_classes=40)
    else:
        model = AttentionModel(**config["model"]["params"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_and_plot(model, csv_file, img_dir, device, transform, class_mapping, config, save_path="results/predictions", file_name="prediction_plot.png"):
    # Load CSV file
    data = pd.read_csv(csv_file)
    
    # Number of images to predict
    num_of_predimg = min(config["prediction"]["num_of_predimg"], 9)
    rows = (num_of_predimg + 2) // 3  # Calculate the number of rows
    cols = min(num_of_predimg, 3)  # Max columns are 3

    # Randomly select images
    random_rows = data.sample(num_of_predimg)
    
    # Prepare the plot
    plt.figure(figsize=(cols * 5, rows * 5))
    
    for i, (_, row) in enumerate(random_rows.iterrows()):
        img_name = row["FileName"]
        img_path = os.path.join(img_dir, img_name)
        
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make predictions
        with torch.no_grad():
            class_logits, binary_logits = model(input_tensor)
            class_idx = torch.argmax(class_logits, dim=1).item()  # Predicted class index
            binary_pred = torch.sigmoid(binary_logits).round().item()  # Binary prediction
        
        # Map class index to class name
        class_label = class_mapping.get(class_idx, "Unknown") if class_mapping else f"Class {class_idx}"
        binary_label = "Yes" if binary_pred else "No"

        # Plot the image and predictions
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Class: {class_label}\nPerson: {binary_label}")
    
    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, file_name)
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches="tight")
    print(f"Prediction plot saved at {full_path}")


if __name__ == "__main__":
    # Define paths and configurations
    # Get the config file path
    config_file = "cfg/" + sys.argv[1]
    config = load_ymal_config(config_file)

    model_path = config["prediction"]["model_path"]
    csv_file = "A1_2024_data/future_data_2024.csv"
    img_dir = "A1_2024_data/Images"

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, config)
    # Prepare the data
    data_loaders = DataHandler(
        csv_file=config["data"]["csv_file"],
        img_dir=config["data"]["img_dir"],
        augmentations=config["data"]["augmentation"]
    )

    class_mapping = data_loaders.dataset.reverse_class_mapping
    # Predict and plot
    predict_and_plot(model, csv_file, img_dir, device, transform,
                      class_mapping, config, save_path="results/predictions", 
                      file_name="prediction_result.png")
