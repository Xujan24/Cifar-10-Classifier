import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('layer1', nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=32),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )),
            ('layer2', nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=64),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )),
            ('fc1', nn.Sequential(
                nn.Linear(in_features=8 * 8 * 64, out_features=1024),
                nn.ReLU()
            ))
        ]))
        self.out = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        for name, module in self.layers.named_children():
            x = module(x)

            if name == 'layer2':
                x = x.view(x.size(0), -1)

        x = self.out(x)
        return F.softmax(x, dim=1)
