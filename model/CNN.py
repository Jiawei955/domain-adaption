import torch
import torch.nn as nn
from torchvision.models import vgg16
import ssl


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class CNNnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            Flatten(),
            # nn.Linear(4608, 120),
            nn.Linear(1152, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        feature = self.feature(x)       # Sequential_1
        pred = self.classifier(feature)       # Dropout -> Dense -> Activation
        return pred, feature


class VGGnet(torch.nn.Module):
    def __init__(self):
        super(VGGnet, self).__init__()
        features = list(vgg16(pretrained=True).features)
        features.append(Flatten())
        features.append(nn.Linear(in_features=25088,out_features=1024,bias=True))
        features.append(nn.Linear(in_features=1024,out_features=128,bias=True))
        self.feature = nn.ModuleList(features).eval()

        self.classifer = nn.Sequential(
            nn.Linear(128,31),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for layer in self.feature:
            x = layer(x)

        pred = self.classifer(x)
        return pred, x





