from data.custom_dataset import CustomDataset
from utils.plotter import TrainingPlotter
from utils.config_util import load_ymal_config
from data.data import DataHandler

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
data_loaders = DataHandler(
    csv_file=config["data"]["csv_file"],
    img_dir=config["data"]["img_dir"],
    augmentations=config["data"]["augmentation"]
)

print(f"{data_loaders.dataset.class_mapping}")