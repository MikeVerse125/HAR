import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=40):
        """
        Multi-task learning model for image classification and binary prediction.

        Args:
            num_classes (int): Number of classes for the classification task.
        """
        super(MultiTaskModel, self).__init__()

        # Shared Backbone: Pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and avgpool layers

        # Adaptive Pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Binary Classification Head (MoreThanOnePerson)
        self.binary_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary output: YES (1), NO (0)
        )

    def forward(self, x):
        """
        Forward pass for the multi-task model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            Tuple: Classification logits and binary prediction logits.
        """
        # Extract shared features
        features = self.backbone(x)
        features = self.global_avg_pool(features)
        features = features.view(features.size(0), -1)  # Flatten the tensor

        # Task-specific predictions
        class_logits = self.classification_head(features)
        binary_logits = self.binary_head(features).squeeze(-1) 

        return class_logits, binary_logits
