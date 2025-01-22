from data.custom_dataset import CustomDataset
from utils.plotter import TrainingPlotter
from utils.config_util import load_ymal_config
from data.data import DataHandler
import torch
from eval import TestEvaluator

import sys
dataset = CustomDataset(csv_file="A1_2024_data/train_data_2024.csv", data_dir="A1_2024_data/Images")
# print(f"Class mapping: {dataset.class_mapping}")
# print(f"Sample label: {dataset[0][1][0]}")


# plotter = TrainingPlotter(log_file="logs/training.csv")
# metrics = ["f1_score", "loss"]
# heads = ["classification", "binary"]
# plotter = TrainingPlotter(log_file="logs/training.csv")
# plotter.plot_and_save_metrics(metrics=metrics, heads=heads, save_path="results/metrics")



if len(sys.argv) != 2:
    raise ValueError("Usage: python main.py <config.yaml>")
 # Get the config file path
config_file = "cfg/" + sys.argv[1]

# Ensure the config file exists


print(f"Using the config file: {config_file}")

# Load YAML configuration file
config = load_ymal_config(config_file)

# Prepare the data
# plotter = TrainingPlotter(log_file="logs/"+config["model"]["type"]+"_history.csv", model_name=config["model"]["type"])
# metrics = ["f1_score", "loss"]
# heads = ["classification", "binary"]
# plotter.plot_and_save_metrics(metrics=metrics, heads=heads, save_path="results/metrics")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load("results/ModelsSaved/baseline.pth")
loaded_model.to(device)
data_loaders = DataHandler(
            csv_file=config["data"]["csv_file"],
            img_dir=config["data"]["img_dir"],
            augmentations=config["data"]["augmentation"]
        )
dataset = data_loaders.get_dataloaders(
    batch_size=config["data"]["batch_size"],
    num_workers=config["data"]["num_workers"],
)

evaluator = TestEvaluator(
            model=loaded_model,                        # Your trained model
            data_loader=dataset,                # Test DataLoader
            device=device,
            class_names=data_loaders.dataset.class_mapping,  # Map class indices to class names
            log_dir="results/predictions"       # Directory to save heatmap images      
        )
results = evaluator.evaluate()
evaluator.plot_combined_heatmap(results, picture_name="final_baseline_heatmap.png")
