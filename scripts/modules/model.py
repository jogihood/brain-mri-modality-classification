import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"]= "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MRI2DCNN(nn.Module):
    def __init__(self, image_size, classes, c1, c2, dropout):
        super(MRI2DCNN, self).__init__()
        h, _ = image_size

        # Layer 1 (conv1)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        
        # Layer 2 (conv2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        
        # Layer 3 (FC)
        self.fc1 = nn.Linear(c2 * (h//4) * (h//4), 1024).to(device)
        
        # self.bn3 = nn.BatchNorm1d(1024).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        # Layer 4 (FC)
        self.fc2 = nn.Linear(1024, classes).to(device)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class MRI3DCNN(nn.Module):
    def __init__(self, image_size, classes, c1, c2, dropout):
        super(MRI3DCNN, self).__init__()
        d, h, w = image_size
        # Layer 1 (conv1)
        self.conv1 = nn.Conv3d(1, c1, kernel_size=3, padding=1).to(device)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2).to(device)
        
        # Layer 2 (conv2)
        self.conv2 = nn.Conv3d(c1, c2, kernel_size=3, padding=1).to(device)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2).to(device)
        
        # Layer 3 (FC)
        self.fc1 = nn.Linear(c2 * (d//4) * (h//4) * (w//4), 1024).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

        # Layer 4 (FC)
        self.fc2 = nn.Linear(1024, classes).to(device)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class MRI2DResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MRI2DResNet, self).__init__()

        # Pretrained weights (for fine-tuning)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # New params for grayscale(1-channel) image
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # new layers
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, 32, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(32 * 7 * 7, 3)
        self.fc2 = nn.Linear(3, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        #new layers
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Flatten before FC layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x