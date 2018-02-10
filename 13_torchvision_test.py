#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-20 下午4:31
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

import torchvision.models as models

if __name__ == "__main__":

    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet_161()