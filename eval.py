import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class TestEvaluator:
    def __init__(self, model, data_loader, device, class_names, log_dir="results/predictions"):
        """
        Initialize the TestEvaluator class.
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.class_names = class_names
        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists

    def evaluate(self):
        """
        Run the model on the test dataset and collect predictions and ground truths.
        """
        self.model.eval()
        class_preds, class_labels = [], []
        binary_preds, binary_labels = [], []

        with torch.no_grad():
            for images, labels in self.data_loader["test"]:
                images = images.to(self.device)
                labels = labels.to(self.device)

                class_logits, binary_logits = self.model(images)

                # Predictions
                class_preds_batch = torch.argmax(class_logits, dim=1).cpu().numpy()
                binary_preds_batch = (torch.sigmoid(binary_logits).round()).cpu().numpy()

                # Ground truth
                class_labels_batch = labels[:, 0].cpu().numpy()
                binary_labels_batch = labels[:, 1].cpu().numpy()

                # Collect results
                class_preds.extend(class_preds_batch)
                class_labels.extend(class_labels_batch)
                binary_preds.extend(binary_preds_batch)
                binary_labels.extend(binary_labels_batch)

        return {
            "class_preds": np.array(class_preds),
            "class_labels": np.array(class_labels),
            "binary_preds": np.array(binary_preds),
            "binary_labels": np.array(binary_labels),
        }

    def plot_combined_heatmap(self, results, picture_name="combined_heatmap.png"):
        """
        Plot and save a combined heatmap for classification and binary results.
        """
        # Classification Confusion Matrix
        class_cm = confusion_matrix(results["class_labels"], results["class_preds"])
        class_cm_normalized = class_cm.astype("float") / class_cm.sum(axis=1)[:, np.newaxis]

        # Binary Confusion Matrix
        binary_cm = confusion_matrix(results["binary_labels"], results["binary_preds"])
        binary_cm_normalized = binary_cm.astype("float") / binary_cm.sum(axis=1)[:, np.newaxis]

        # Plot combined heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Classification Heatmap
        sns.heatmap(
            class_cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0],
        )
        axes[0].set_title("Classification Confusion Matrix")
        axes[0].set_xlabel("Predicted Labels")
        axes[0].set_ylabel("True Labels")

        # Binary Heatmap
        sns.heatmap(
            binary_cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=axes[1],
        )
        axes[1].set_title("Binary Confusion Matrix")
        axes[1].set_xlabel("Predicted Labels")
        axes[1].set_ylabel("True Labels")

        # Save the combined heatmap
        save_path = os.path.join(self.log_dir, picture_name)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved at: {save_path}")
