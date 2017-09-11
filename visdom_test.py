#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-11 下午3:42
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

import visdom
import numpy as np

if __name__ == "__main__":


    vis = visdom.Visdom()
    vis.text('Hello, world!')
    vis.image(np.ones((3, 10, 10)))