#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-11 下午3:19
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

import torch
from torch.autograd import Variable

if __name__ == "__main__":
    v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
    h = v.register_hook(lambda grad: grad * 2)  # double the gradient
    v.backward(torch.Tensor([1, 1, 1]))
    # 先计算原始梯度，再进hook，获得一个新梯度。
    print(v.grad.data)
    h.remove()  # removes the hook
