#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-8 下午8:14
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

import torch

from torch.autograd import Variable

if __name__ == "__main__":
    x = Variable(torch.ones(2,2), requires_grad = True)
    #x = Variable(torch.ones(2, 2))
    print x
    y = x + 2
    print y
    print "y", y.creator, y.data, y.requires_grad

    z = y * y *3
    out = z.mean()
    print "z, out", z, out
    out.backward()
    print "x grad", x.grad, x.data
    print "y grad", y.grad

    print "==========2"
    x = torch.randn(3)
    print x
    x = Variable(x, requires_grad=True)
    y = x * 2

    print y.data
    while y.data.norm() < 1000:
        y = y * 2

    print(y)
    print y.creator.num_outputs, y.output_nr
    initial_grad = [None for _ in range(y.creator.num_outputs)]
    print "init_grad", initial_grad
    gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
    y.backward(gradients)
    print x.grad
    print "==========2 end"


    print "==========3"
    a = Variable(torch.FloatTensor([2, 3]), requires_grad = True)
    b = a + 3
    c = b*b*3
    out = c.mean()
    out.backward()
    print "input", a.data
    print "compute result is", out.data, out.data[0]
    print "input gradient are", a.grad.data