import torch
import torch.nn as nn
from torchvision.transforms import Resize

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer0 = Resize((768, 1366), antialias=True)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=0
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0
            ),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=32, out_channels=64, kernel_size=7, stride=5, padding=1
        #     ),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0
            ),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0
            ),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(3072, 2),
            # nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Dropout(0.2),
        )
        # self.layer8 = nn.Sequential(nn.Linear(256, 2))

    def forward(self, X):
        out = self.layer0(X)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer7(out)
        return out
