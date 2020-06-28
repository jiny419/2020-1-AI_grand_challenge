#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class Classifier_mrcg(nn.Module):
    def __init__(self):
        super(Classifier_mrcg, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (9,7), padding=0), # 768, 100 -> 760, 94
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (9,7), padding=0), # 760, 94-> 752, 88
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((3,2)) # 752, 88 -> 250,44
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (9,7), padding=0), #250, 44 -> 242, 38
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32,(9,7), padding=0), #242, 38 -> 234, 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((3,2)) #234, 32 -> 78, 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (9,7), padding=0), #78, 16 -> 70, 10
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((3,2)), # 70, 10 -> 23, 5
            nn.Conv2d(64, 64, 3, padding=1), # (batch, 64, 23, 5)
            nn.ReLU()

        )
        self.fc_module = nn.Sequential(
            nn.Linear(64*23*5, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 1)

        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(x.shape[0], -1) # 7360 (batch, 7360)
        out = self.fc_module(out)

        return out