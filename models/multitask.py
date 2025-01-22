from torch import nn
import torch
import torch.nn.functional  as F 
from torchvision.models import resnet50, mobilenet_v3_small
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

# Determine whichs features map are important for current input tensor 
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected network
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_feats = self.global_avg_pool(x)
        avg_feats = self.global_max_pool(x)

        max_feats = torch.flatten(max_feats, 1)
        avg_feats = torch.flatten(avg_feats, 1)

        max_feats = self.fc(max_feats)
        avg_feats = self.fc(avg_feats)

        output = (
            self.sigmoid(max_feats + avg_feats).unsqueeze(2).unsqueeze(3).expand_as(x)
        )

        return output * x

# Focus on where(spatial location) is the most relevent feature in the feature map
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average Pooling  along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max Pooling along the channel axis
        max_out,_ = torch.max(x, dim=1, keepdim=True)

        # Concate and apply the convolution 
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        out = self.sigmoid(out)
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)


    def forward(self, x):
        # Apply Channel Attention
        x = self.channel_attention(x)

        # Apply Spatial Attention
        x = self.spatial_attention(x)

        return x

class AttentionModel(nn.Module):
    def __init__(self, reduction_ratio, kernel_size, finetune, dropout_rate=0.5, in_channels=[256, 128, 32]):
        super().__init__()

        # Action Backbone: ResNet50 with CBAM
        self.action_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        if not finetune:
            for param in self.action_backbone.parameters():
                param.requires_grad = False
            for param in list(self.action_backbone.children())[-3:]:  # Unfreeze deeper layers
                for p in param.parameters():
                    p.requires_grad = True
        action_n_feats = self.action_backbone.fc.in_features
        # Remove fully connected layer
        self.action_backbone.fc = nn.Identity()
        # Retain only convolutional layer
        self.action_backbone = nn.Sequential(*list(self.action_backbone.children())[:-2])
        self.action_att = CBAM(action_n_feats, reduction_ratio, kernel_size)
        
        # Person Backbone: MobileNetV3-Small with Spatial Attention
        self.person_backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        if not finetune:
            for param in self.person_backbone.parameters():
                param.requires_grad = True
        person_n_feats = self.person_backbone.classifier[0].in_features
        self.person_backbone.classifier = nn.Identity()
        self.person_backbone = nn.Sequential(*list(self.person_backbone.children())[:-2])
        self.person_att = SpatialAttention(kernel_size)
        
        # Pooling layers
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.action_fc = nn.Sequential(
            nn.Linear(action_n_feats, in_channels[0]),
            nn.BatchNorm1d(in_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels[0], in_channels[1]),
            nn.BatchNorm1d(in_channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.person_fc = nn.Sequential(
            nn.Linear(person_n_feats, in_channels[2]),
            nn.BatchNorm1d(in_channels[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Output Layers
        self.action_out = nn.Linear(in_channels[1], 40)  # 40 classes
        self.person_out = nn.Linear(in_channels[2], 1)  # Binary classification

    def forward(self, x):
        # Action Classification
        action = self.action_backbone(x)
        action = self.action_att(action)
        action = torch.flatten(self.pool_1(action), 1)
        action = self.action_fc(action)
        action = self.action_out(action)

        # Binary Classification
        person = self.person_backbone(x)
        person = self.person_att(person)
        person = torch.flatten(self.pool_2(person), 1)
        person = self.person_fc(person)
        person = self.person_out(person).squeeze(-1)

        return action, person