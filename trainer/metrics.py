from torchmetrics import F1Score, Precision, Recall


class Metrics:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes

        # Classification Metrics
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)

        # Binary Metrics
        self.f1_binary = F1Score(task="binary").to(device)
        self.precision_binary = Precision(task="binary").to(device)
        self.recall_binary = Recall(task="binary").to(device)

    def update(self, class_preds, class_labels, binary_preds, binary_labels):
    
        # Update classification metrics
        self.f1.update(class_preds, class_labels)
        self.precision.update(class_preds, class_labels)
        self.recall.update(class_preds, class_labels)

        # Update binary metrics
        self.f1_binary.update(binary_preds, binary_labels)
        self.precision_binary.update(binary_preds, binary_labels)
        self.recall_binary.update(binary_preds, binary_labels)

    def compute(self):
        """
        Compute the final metrics values.
        """
        return {
            "classification_f1_score": self.f1.compute().item(),
            "classification_precision": self.precision.compute().item(),
            "classification_recall": self.recall.compute().item(),
            "binary_f1_score": self.f1_binary.compute().item(),
            "binary_precision": self.precision_binary.compute().item(),
            "binary_recall": self.recall_binary.compute().item()
        }

    def reset(self):
        """Reset all the metrics states"""
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

        self.f1_binary.reset()
        self.precision_binary.reset()
        self.recall_binary.reset()
