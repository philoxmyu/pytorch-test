#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-11 下午3:15
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com


'''
调用 forward 方法计算结果
判断有没有注册 forward_hook，有的话，就将 forward 的输入及结果作为hook的实参。然后让hook自己干一些不可告人的事情
'''
import torch
from torch import nn
import torch.functional as F
from torch.autograd import Variable


def for_hook(module, input, output):
    print(module)
    for val in input:
        print("input val:", val)
    for out_val in output:
        print("output val:", out_val)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x + 1

if __name__ == "__main__":


    model = Model()
    x = Variable(torch.FloatTensor([1]), requires_grad=True)
    handle = model.register_forward_hook(for_hook)
    print(model(x))
    handle.remove()
