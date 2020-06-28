#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class Classifier_stft(nn.Module):
    def __init__(self):
        super(Classifier_stft, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, (9, 7), padding=0),  # 512, 100 -> 504, 94
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (9, 7), padding=0),  # 504, 94 -> 496, 88
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((3, 2))  # 496, 88 -> 165, 44
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (9, 7), padding=0),  # 165, 44 -> 157, 38
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (9, 7), padding=0),  # 157, 38 -> 149, 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((3, 2))  # 149, 32 -> 49, 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (9, 7), padding=0),  # 49, 16-> 41, 10
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((3, 2)),  # 41, 10-> 13, 5 
            nn.Conv2d(64, 64, 3, padding=1),  # same padding (bath, 64, 13, 5) 
            nn.ReLU()

        )
        self.fc_module = nn.Sequential(
            nn.Linear(64 * 13 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(x.shape[0], -1)  # 4160 (batch, 4160)
        out = self.fc_module(out)

        return out