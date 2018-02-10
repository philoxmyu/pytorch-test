#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-8 下午7:46
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization

if __name__ == "__main__":
    cnn = CNN()
    print(cnn)  # net architecture

    print cnn.conv1

    # childrens = cnn.named_children()
    # for name, item in childrens:
    #     print name, item, type(item)
    #
    # print "*****************************"
    # for name, item in cnn.named_modules():
    #     print name, item, type(item)
    #
    # print "*****************************"
    for name, item in cnn.named_parameters():
        print name

    print "************"
    print "cnn parameters:", cnn.parameters()
    for item in cnn.parameters():
        print "param:", type(item)
    params = list(cnn.parameters())
    print(len(params))
    print(params[0].size())  # conv1.0's .weight
    print(params[1].size())  # conv1.0's bias