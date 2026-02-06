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

        # Learning rate scheduler (optional)
        self.use_scheduler = self.config["training"].get("use_scheduler", False)
        if self.use_scheduler:
            scheduler_config = self.config["training"].get("scheduler", {})
            scheduler_type = scheduler_config.get("type", "ReduceLROnPlateau")
            
            if scheduler_type == "ReduceLROnPlateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=scheduler_config.get("mode", "min"),
                    factor=scheduler_config.get("factor", 0.5),
                    patience=scheduler_config.get("patience", 3)
                )
            elif scheduler_type == "CosineAnnealingLR":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get("T_max", self.config["training"]["epochs"]),
                    eta_min=scheduler_config.get("eta_min", 1e-6)
                )
            elif scheduler_type == "StepLR":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get("step_size", 10),
                    gamma=scheduler_config.get("gamma", 0.1)
                )
            else:
                self.scheduler = None
                print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
        else:
            self.scheduler = None

        # Task-specific loss weights
        self.classification_weight = self.config["training"].get("classification_weight", 1.0)
        self.binary_weight = self.config["training"].get("binary_weight", 1.0)

        # Logs
        self.log_file = log_file

        # Initialize the history for storing metrics (optional, only if needed for in-memory analysis)
        self.track_history = self.config["training"].get("track_history", False)
        self.history = {"epoch": []} if self.track_history else None

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
            binary_preds = torch.sigmoid(binary_logits).round().int()
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
                binary_loss = self.criterion_binary(binary_logits, binary_labels)

                # Accumulate losses and update metrics
                total_classification_loss += classification_loss.item()
                total_binary_loss += binary_loss.item()

                class_preds = torch.argmax(class_logits, dim=1)
                binary_preds = torch.sigmoid(binary_logits).round().int()
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
        The best model (lowest validation loss) is always saved and clearly labeled.
        """
        # Validate dataloaders
        if len(self.dataloaders["train"]) == 0:
            raise ValueError("Training dataloader is empty. Cannot proceed with training.")
        if len(self.dataloaders["val"]) == 0:
            raise ValueError("Validation dataloader is empty. Cannot proceed with training.")
        
        # Verify GPU usage
        print(f"Training on device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        total_epochs = self.config["training"]["epochs"]
        early_stopping = self.config["training"]["early_stopping"]
        patience = self.config["training"]["patience"]

        best_metric = float("inf")  # Assuming we are minimizing validation loss
        patience_counter = 0
        best_epoch = -1

        for epoch in range(total_epochs):
            start_time = time.time()  # Record start time for the epoch

            # Train and evaluate
            train_metrics = self.__train()
            eval_metrics = self.__evaluate()

            # Track the evaluation loss (or other monitored metric)
            eval_loss = eval_metrics["classification_loss"] + eval_metrics["binary_loss"]

            # Update learning rate scheduler
            if self.use_scheduler and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()

            # Update best metric and reset patience counter
            if eval_loss < best_metric:
                best_metric = eval_loss
                patience_counter = 0
                best_epoch = epoch + 1

                # Save the best model with clear naming
                if self.config["training"]["save_model"]:
                    best_model_name = f"best_{self.model_name}"
                    self.__save_model(model_name=best_model_name)
                    print(f"Best model saved at epoch {epoch + 1} with validation loss: {eval_loss:.4f}")
            else:
                patience_counter += 1

            # Log metrics to history (only if tracking is enabled)
            if self.track_history:
                self.history["epoch"].append(epoch + 1)
                for key, value in train_metrics.items():
                    if f"train_{key}" not in self.history:
                        self.history[f"train_{key}"] = []
                    self.history[f"train_{key}"].append(value)
                for key, value in eval_metrics.items():
                    if f"eval_{key}" not in self.history:
                        self.history[f"eval_{key}"] = []
                    self.history[f"eval_{key}"].append(value)

            # Write metrics to CSV
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1] + list(train_metrics.values()) + list(eval_metrics.values()))

            # Calculate epoch time
            epoch_time = time.time() - start_time
            estimated_time_remaining = epoch_time * (total_epochs - epoch - 1)

            # Print metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch + 1}/{total_epochs}]-------------------------------------------------------------")
            print(f"Train: {train_metrics}")
            print(f"Eval: {eval_metrics}")
            print(f"Eval Loss: {eval_loss:.4f}, Best Eval Loss: {best_metric:.4f}, Patience Counter: {patience_counter}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Epoch Time: {epoch_time:.2f}s, Estimated Time Remaining: {estimated_time_remaining / 60:.2f} minutes")

            # Early stopping
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best model was at epoch {best_epoch}.")
                break

        print("---------------------Training Complete---------------------")
        if best_epoch > 0:
            print(f"Best model achieved at epoch {best_epoch} with validation loss: {best_metric:.4f}")
            if self.config["training"]["save_model"]:
                print(f"Best model saved as: best_{self.model_name}")
        else:
            print("Warning: No improvement was observed during training.")


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