from torch import nn
import torch
import torch.nn as nn
import torchvision.models as vision_models

class CNNNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    # 4 conv blocks / flatten / linear / softmax
    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=2
        ),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=2
        ),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=2
        ),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=2
        ),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.flatten = nn.Flatten()
    self.linear = nn.Linear(128 * 5 * 4, 12)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, input_data):
    x = self.conv1(input_data)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)   
    logits = self.linear(x)
    predictions = self.softmax(logits)
    return predictions
  
class ExtedndedCNNNetwork(CNNNetwork):
    def __init__(self, num_classes=12):
        super(CNNNetwork, self).__init__()
        
        # Load a pretrained ResNet model
        self.pretrained_model = vision_models.resnet18(weights=vision_models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the input layer to accept single-channel input
        self.pretrained_model.conv1 = nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=self.pretrained_model.conv1.out_channels,
            kernel_size=self.pretrained_model.conv1.kernel_size,
            stride=self.pretrained_model.conv1.stride,
            padding=self.pretrained_model.conv1.padding,
            bias=False
        )
        
        # Freeze the pretrained layers
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Modify the last fully connected layer to match the number of classes
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(self.pretrained_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input_data):
        x = self.pretrained_model(input_data)
        return x

if __name__ == "__main__":
  # Initialize the model
  num_classes = 12
  model = ExtedndedCNNNetwork(num_classes)

  # Move the model to GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)