from transformers import ResNetConfig, ResNetForImageClassification, Dinov2WithRegistersForImageClassification
import torch
import numpy as np
from constants import *

def get_tinyHomemadeCNN(num_labels=None):
    class TinyHomemadeCNN(torch.nn.Module):
        def __init__(self, num_labels=3):
            super(TinyHomemadeCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = torch.nn.Linear(32 * 56 * 56, 128)
            self.fc2 = torch.nn.Linear(128, num_labels)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 32 * 56 * 56)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    if num_labels is not None:
        model = TinyHomemadeCNN(num_labels=num_labels)
    else:
        model = TinyHomemadeCNN()
    return model

def get_dinov2(num_labels=None):
    if num_labels is not None:
        model = Dinov2WithRegistersForImageClassification.from_pretrained(
            "facebook/dinov2-with-registers-base",
            num_labels=num_labels,
        )
    else:
        model = Dinov2WithRegistersForImageClassification.from_pretrained("facebook/dinov2-with-registers-base")
    return model
    
def get_resnet18(num_labels=None, checkpoint_path=None):
    configuration = ResNetConfig(
        depths=[2, 2, 2, 2],
        downsample_in_bottleneck=False,
        downsample_in_first_stage=False,
        embedding_size=64,
        hidden_act="relu",
        hidden_sizes=[64, 128, 256, 512],
        layer_type="basic",
        model_type="resnet",
        num_channels=3,
        out_features=["stage4"],
        out_indices=[4],
        stage_names=["stem", "stage1", "stage2", "stage3", "stage4"]
    )
    if num_labels is not None:
        configuration.num_labels = num_labels
    model = ResNetForImageClassification(configuration)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    return model