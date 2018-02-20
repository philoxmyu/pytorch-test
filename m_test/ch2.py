#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-22 下午2:54
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

from __future__ import print_function
import torch as t

if __name__ == "__main__":
    x = t.Tensor(5, 3)
    print(x)
    print(x.size())

    y = t.rand(5,3)
    print(y)
    print(x+y) # t.add(x,y)

    # 加法的第三种写法：指定加法结果的输出目标为result
    result = t.Tensor(5, 3)  # 预先分配空间
    t.add(x, y, out=result)  # 输入到result
    print(result)

    #########################3
    # 注意，函数名后面带下划线**`_`** 的函数会修改Tensor本身。例如，`x.add_(y)`和`x.t_()`会改变 `x`，但`x.add(y)`和`x.t()`返回一个新的Tensor， 而`x`不变。
    print('最初y')
    print(y)

    print('第一种加法，y的结果')
    y.add(x)  # 普通加法，不改变y的内容
    print(y)

    print('第二种加法，y的结果')
    y.add_(x)  # inplace 加法，y变了
    print(y)

    print("*" * 10)
    a = t.ones(5)
    print(a)
    print(a.size())  # (5,)

    b = a.numpy()  # tensor to numpy
    print(b, type(b))

    # numpy to tensor
    import numpy as np
    a = np.ones(5) # (5, )
    b = t.from_numpy(a)  # Numpy->Tensor
    print(a)
    print(b)
    # Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变

    b.add_(1)  # 以`_`结尾的函数会修改自身
    print(a)
    print(b)  # Tensor和Numpy共享内存

    #Tensor可通过 `.cuda` 方法转为GPU的Tensor，从而享受GPU带来的加速运算。
    if t.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        print(x + y)


    # 深度学习的算法本质上是通过反向传播求导数，而PyTorch的**`Autograd`**模块则实现了此功能。在Tensor上的所有操作，Autograd都能为它们自动提供微分，避免了手动计算导数的复杂过程。
    # `autograd.Variable`是Autograd中的核心类，它简单封装了Tensor，并支持几乎所有Tensor有的操作。
    # Tensor在被封装为Variable之后，可以调用它的`.backward`实现反向传播，自动计算所有梯度。Variable的数据结构如图2-6所示。
    print("* autograd" * 5)
    from torch.autograd import Variable
    x = Variable(t.ones(2, 2), requires_grad=True)
    print(x, x.grad, x.data)
    y = x.sum()
    print("y:", y)
    print("y.grad_fn:", y.grad_fn) # <SumBackward0 object at 0x7fa07e8683d0>
    y.backward()
    # y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
    # 每个值的梯度都为1
    print("x.grad:", x.grad)
    # 注意：`grad`在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零。
    y.backward()
    print("x.grad2:", x.grad)

    # 以下划线结束的函数是inplace操作，就像add_
    x.grad.data.zero_()
    y.backward()
    print("x.grad3:", x.grad)

    #Variable和Tensor具有近乎一致的接口，在实际使用中可以无缝切换。
    x = Variable(t.ones(4, 5))
    print("x data type:", x.data)
    y = t.cos(x)
    x_tensor_cos = t.cos(x.data)
    print(y, x_tensor_cos)