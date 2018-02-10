#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-11 上午10:53
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

'''
关于求梯度，只有我们定义的Variable才会被求梯度，由creator创造的不会去求梯度
自己定义Variable的时候，记得Variable(Tensor, requires_grad = True),这样才会被求梯度，不然的话，是不会求梯度的
'''

import torch
from torch.autograd import Variable

if __name__ == "__main__":

    x = torch.rand(5)
    x = Variable(x, requires_grad=True)
    y = x * 2
    grads = torch.FloatTensor([1, 2, 3, 4, 5])
    y.backward(grads)  # 如果y是scalar的话，那么直接y.backward()，然后通过x.grad方式，就可以得到var的梯度
    print x.grad  # 如果y不是scalar，那么只能通过传参的方式给x指定梯度\
    print y.grad # None

    # numpy to Tensor
    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)  # 如果a 变的话， b也会跟着变，说明b只是保存了一个地址而已，并没有深拷贝
    print(b)  # Variable只是保存Tensor的地址，如果Tensor变的话，Variable也会跟着变

    print "----------------------"

    a = np.ones(5)
    b = torch.from_numpy(a)  # ndarray --> Tensor
    a_ = b.numpy()  # Tensor --> ndarray
    np.add(a, 1, out=a)  # 这个和 a = np.add(a,1)有什么区别呢？
    # a = np.add(a,1) 只是将a中保存的指针指向新计算好的数据上去
    # np.add(a, 1, out=a) 改变了a指向的数据
    print a,b
    print a_

    # tensor 与 numpy
    n1 = np.array([1., 2.]).astype(np.float32)
    t1 = torch.FloatTensor(n1)
    #t1 = torch.from_numpy(n1)
    n1[0] = 2.
    print(t1)     # 可以看出，当使用 无论是使用 FloatTensor 还是 from_numpy 来创建 tensor
                  # tensor 只是指向了 初始的值而已，而没有自己再开辟空间。
                  # FloatTensor(2,3,2) 这个不一样，它是开辟了一个 空间。
