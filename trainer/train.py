import torch
import csv
import os
import time


class Trainer:
    def __init__(self, model, dataLoaders, config, device, metrics, model_name, log_file="training.csv"):
        """
        Initialize the trainer class.
        """
        self.model = model
        self.model_name = model_name
        self.dataloaders = dataLoaders
        self.config = config
        self.device = device
        self.metrics = metrics

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training"]["learning_rate"])
        self.criterion_classification = torch.nn.CrossEntropyLoss()
        self.criterion_binary = torch.nn.BCEWithLogitsLoss()

        # Task-specific loss weights
        self.classification_weight = self.config.get("classification_weight", 1.0)
        self.binary_weight = self.config.get("binary_weight", 1.0)

        # Logs
        self.log_file = log_file

        # Initialize the history for storing metrics
        self.history = {"epoch": []}

        # Create log file if it does not exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as file:
                writer = csv.writer(file)
                header = ["epoch"]
                for phase in ["train", "eval"]:
                    for metric in [
                        "classification_loss",
                        "binary_loss",
                        "classification_f1_score",
                        "classification_precision",
                        "classification_recall",
                        "binary_f1_score",
                        "binary_precision",
                        "binary_recall",
                        "total_loss",
                    ]:
                        header.append(f"{phase}_{metric}")
                writer.writerow(header)

    def __train(self):
        """
        Private method to perform one epoch of training.
        """
        self.model.train()
        total_classification_loss = 0.0
        total_binary_loss = 0.0

        self.metrics.reset()
        for image, labels in self.dataloaders["train"]:
            image = image.to(self.device)
            labels = labels.to(self.device)

            class_labels = labels[:, 0].long()
            binary_labels = labels[:, 1].float()

            self.optimizer.zero_grad()

            # Forward pass
            class_logits, binary_logits = self.model(image)

            # Compute the loss
            classification_loss = self.criterion_classification(class_logits, class_labels)
            binary_loss = self.criterion_binary(binary_logits, binary_labels)
            loss = self.classification_weight * classification_loss + self.binary_weight * binary_loss

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate losses and update metrics
            total_classification_loss += classification_loss.item()
            total_binary_loss += binary_loss.item()

            class_preds = torch.argmax(class_logits, dim=1)
            binary_preds = torch.sigmoid(binary_logits).round()
            self.metrics.update(class_preds, class_labels, binary_preds, binary_labels)

        # Compute average losses and metrics
        avg_classification_loss = total_classification_loss / len(self.dataloaders["train"])
        avg_binary_loss = total_binary_loss / len(self.dataloaders["train"])
        total_loss = self.classification_weight * avg_classification_loss + self.binary_weight * avg_binary_loss
        metrics = self.metrics.compute()

        return {
            "classification_loss": avg_classification_loss,
            "binary_loss": avg_binary_loss,
            "classification_f1_score": metrics["classification_f1_score"],
            "classification_precision": metrics["classification_precision"],
            "classification_recall": metrics["classification_recall"],
            "binary_f1_score": metrics["binary_f1_score"],
            "binary_precision": metrics["binary_precision"],
            "binary_recall": metrics["binary_recall"],
            "total_loss": total_loss,
        }

    def __evaluate(self):
        """
        Private method to perform evaluation.
        """
        self.model.eval()
        total_classification_loss = 0.0
        total_binary_loss = 0.0

        self.metrics.reset()
        with torch.no_grad():
            for image, labels in self.dataloaders["val"]:
                image = image.to(self.device)
                labels = labels.to(self.device)

                class_labels = labels[:, 0].long()
                binary_labels = labels[:, 1].float()
                
                # Forward pass
                class_logits, binary_logits = self.model(image)
                
                # Compute the loss
                classification_loss = self.criterion_classification(class_logits, class_labels)
                binary_loss = self.criterion_binary(binary_logits, binary_labels.float())

                # Accumulate losses and update metrics
                total_classification_loss += classification_loss.item()
                total_binary_loss += binary_loss.item()

                class_preds = torch.argmax(class_logits, dim=1)
                binary_preds = torch.sigmoid(binary_logits).round()
                self.metrics.update(class_preds, class_labels, binary_preds, binary_labels)

        # Compute average losses and metrics
        avg_classification_loss = total_classification_loss / len(self.dataloaders["val"])
        avg_binary_loss = total_binary_loss / len(self.dataloaders["val"])
        total_loss = self.classification_weight * avg_classification_loss + self.binary_weight * avg_binary_loss
        metrics = self.metrics.compute()

        return {
            "classification_loss": avg_classification_loss,
            "binary_loss": avg_binary_loss,
            "classification_f1_score": metrics["classification_f1_score"],
            "classification_precision": metrics["classification_precision"],
            "classification_recall": metrics["classification_recall"],
            "binary_f1_score": metrics["binary_f1_score"],
            "binary_precision": metrics["binary_precision"],
            "binary_recall": metrics["binary_recall"],
            "total_loss": total_loss,
        }


    def fit(self):
        """
        Train and evaluate the model with early stopping.
        Save the model in two cases:
        1. After 5 patience epochs without improvement.
        2. At the end of training if early stopping is not triggered.
        """
        total_epochs = self.config["training"]["epochs"]
        early_stopping = self.config["training"]["early_stopping"]
        patience = self.config["training"]["patience"]

        best_metric = float("inf")  # Assuming we are minimizing validation loss
        patience_counter = 0
        best_epoch = -1
        model_saved = False  # Track if the model has been saved

        for epoch in range(total_epochs):
            start_time = time.time()  # Record start time for the epoch

            # Train and evaluate
            train_metrics = self.__train()
            eval_metrics = self.__evaluate()

            # Track the evaluation loss (or other monitored metric)
            eval_loss = eval_metrics["classification_loss"] + eval_metrics["binary_loss"]

            # Update best metric and reset patience counter
            if eval_loss < best_metric:
                best_metric = eval_loss
                patience_counter = 0
                best_epoch = epoch + 1

                # Save the best model
                if self.config["training"]["save_model"]:
                    self.__save_model(model_name=self.model_name)
                    model_saved = True
                    print(f"Best model saved at epoch {epoch + 1}.")
            else:
                patience_counter += 1

                # Save the model after 5 patience epochs if it's not improving
                if patience_counter == patience and self.config["training"]["save_model"]:
                    model_name = f"model_after_{patience}_patience_epochs.pth"
                    self.__save_model(model_name=model_name)
                    model_saved = True
                    print(f"Model saved after {patience} patience epochs at epoch {epoch + 1}.")

            # Log metrics to history
            self.history["epoch"].append(epoch + 1)
            for key, value in train_metrics.items():
                self.history[f"train_{key}"] = self.history.get(f"train_{key}", []) + [value]
            for key, value in eval_metrics.items():
                self.history[f"eval_{key}"] = self.history.get(f"eval_{key}", []) + [value]

            # Write metrics to CSV
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1] + list(train_metrics.values()) + list(eval_metrics.values()))

            # Calculate epoch time
            epoch_time = time.time() - start_time
            estimated_time_remaining = epoch_time * (total_epochs - epoch - 1)

            # Print metrics
            print(f"[Epoch {epoch + 1}/{total_epochs}]-------------------------------------------------------------")
            print(f"Train: {train_metrics}")
            print(f"Eval: {eval_metrics}")
            print(f"Eval Loss: {eval_loss:.4f}, Best Eval Loss: {best_metric:.4f}, Patience Counter: {patience_counter}")
            print(f"Epoch Time: {epoch_time:.2f}s, Estimated Time Remaining: {estimated_time_remaining / 60:.2f} minutes")

            # Early stopping
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best model was at epoch {best_epoch}.")
                break

        # Save the model at the end of training if it hasn't already been saved
        if not model_saved and self.config["training"]["save_model"]:
            self.__save_model(model_name="final_" + self.model_name + ".pth")
            print("Final model saved at the end of training.")

        print("---------------------Training Complete---------------------")


    def __save_model(self, model_name="model.pth", save_dir="results/ModelsSaved"):
        """
        Save the trained model to the specified directory.
        """
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Full path to save the model
        save_path = os.path.join(save_dir, model_name)

        # Save the model state dictionary
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")