import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from sklearn.metrics import confusion_matrix
import csv
import os

class TrainingPlotter:
    def __init__(self, log_file, model_name="model"):
        """
        Initialize the plotter with the log file containing training history.
        Args:
            log_file (str): Path to the CSV log file.
        """
        self.log_file = log_file
        self.model_name = model_name

    def plot_and_save_metrics(self, metrics, heads, save_path="results/metrics"):
        """
        Plot and save line charts for each metric and task head combination.
        """
        # Load training history
        history = self.__load_training_history()

        # Ensure save_path directory exists
        os.makedirs(save_path, exist_ok=True)

        # Iterate over each metric and head combination
        for metric in metrics:
            for head in heads:
                # Construct keys for train and eval data
                train_key = f"train_{head.lower()}_{metric.lower()}"
                eval_key = f"eval_{head.lower()}_{metric.lower()}"

                # Check if the keys exist in history
                if train_key not in history or eval_key not in history:
                    print(f"Skipping {metric} for {head}: Data not found in history.")
                    continue

                # Plot data
                plt.figure(figsize=(10, 6))
                plt.plot(history["epoch"], history[train_key], label="Train", color="blue")
                plt.plot(history["epoch"], history[eval_key], label="Eval", color="orange")
                plt.title(f"{head.title()} {metric.replace('_', ' ').title()} Over Epochs")
                plt.xlabel("Epochs")
                plt.ylabel(metric.replace('_', ' ').title())
                plt.legend()
                plt.grid(True)

                # Save the plot
                file_name = f"{self.model_name}_{head.lower()}_{metric.lower()}.png"
                full_path = os.path.join(save_path, file_name)
                plt.savefig(full_path, bbox_inches="tight")
                plt.close()  # Close the plot to free memory
                print(f"Plot saved: {full_path}")

    def __load_training_history(self):
        """
        Load training history from the log file.
        """
        history = {}
        with open(self.log_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    if key not in history:
                        history[key] = []
                    history[key].append(float(value))
        return history
