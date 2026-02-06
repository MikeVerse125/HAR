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
        
        # Convert class_names dict to sorted list by index
        if isinstance(class_names, dict):
            # Sort by value (index) to get correct order: [class_0, class_1, ...]
            self.class_names = [k for k, v in sorted(class_names.items(), key=lambda x: x[1])]
        else:
            self.class_names = class_names
            
        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists

    def evaluate(self):
        """
        Run the model on the test dataset and collect predictions and ground truths.
        """
        # Validate test dataloader exists and is not empty
        if "test" not in self.data_loader:
            raise ValueError("Test dataloader not found in data_loader dictionary.")
        if len(self.data_loader["test"]) == 0:
            raise ValueError("Test dataloader is empty. Cannot evaluate.")
        
        self.model.eval()
        class_preds, class_labels = [], []
        binary_preds, binary_labels = [], []

        with torch.no_grad(): # Do not track gradients during evaluation
            for images, labels in self.data_loader["test"]: # using "test" dataloader
                images = images.to(self.device)
                labels = labels.to(self.device)

                class_logits, binary_logits = self.model(images)

                # Predictions
                class_preds_batch = torch.argmax(class_logits, dim=1).cpu().numpy()
                binary_preds_batch = torch.sigmoid(binary_logits).round().int().cpu().numpy()

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
        # Validate results
        if len(results["class_labels"]) == 0:
            print("Warning: No test samples to plot. Skipping heatmap generation.")
            return
        
        # Get all possible class labels
        num_classes = len(self.class_names)
        labels_range = list(range(num_classes))
        
        # Classification Confusion Matrix with all labels
        # Without this, we can get mismatched with self.class_names 
        class_cm = confusion_matrix(
            results["class_labels"],  # true labels
            results["class_preds"],   # predicted labels
            labels=labels_range       # Force all classes to appear [0,1,2,...,num_classes-1]
        ) 
        # Normalize with NaN handling
        with np.errstate(divide='ignore', invalid='ignore'):
            class_cm_normalized = class_cm.astype("float") / class_cm.sum(axis=1)[:, np.newaxis]
            class_cm_normalized = np.nan_to_num(class_cm_normalized, nan=0.0) # Replace naN with 0.0

        # Binary Confusion Matrix
        binary_cm = confusion_matrix(results["binary_labels"], results["binary_preds"])
        with np.errstate(divide='ignore', invalid='ignore'): 
            binary_cm_normalized = binary_cm.astype("float") / binary_cm.sum(axis=1)[:, np.newaxis]
            binary_cm_normalized = np.nan_to_num(binary_cm_normalized, nan=0.0) # Replace naN with 0.0

        # Adjust figure size based on number of classes
        class_fig_size = max(12, num_classes * 0.4)  # Scale with number of classes
        fig, axes = plt.subplots(1, 2, figsize=(class_fig_size + 8, class_fig_size * 0.5))

        # Classification Heatmap - disable annotations if too many classes
        show_annot = num_classes <= 20  # Only show numbers if 20 or fewer classes
        sns.heatmap(
            class_cm_normalized, 
            annot=show_annot,
            fmt=".2f" if show_annot else "",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0],
            cbar_kws={'label': 'Normalized Count'}
        )
        axes[0].set_title("Classification Confusion Matrix (Normalized)")
        axes[0].set_xlabel("Predicted Labels")
        axes[0].set_ylabel("True Labels")
        # Rotate labels for better readability
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, ha='right')
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

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
