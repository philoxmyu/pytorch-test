#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-11 下午2:41
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

import torch
from torch.autograd import Variable

'''
Variable的一些运算，实际上就是里面的Tensor的运算。 
pytorch中的所有运算都是基于Tensor的，Variable只是一个Wrapper，Variable的计算的实质就是里面的Tensor在计算。Variable默认代表的是里面存储的Tensor（weights）
'''

if __name__ == "__main__":
    w1 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True)  # 需要求导的话，requires_grad=True属性是必须的。
    w2 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True)
    print(w1.grad)  # 0.2 版本打印的是 None
    print(w2.grad)  # 0.2 版本打印的是 None

    # d.backward()求Variable的梯度的时候，Variable.grad是累加的即: Variable.grad=Variable.grad+new_grad
    d = torch.mean(w1)
    d.backward()
    print type(w1), type(w1.grad), type(w1.data), w1.grad

    d.backward()
    print w1.grad

    w1.grad.data.zero_()   #梯度置零
    print w1.grad

    # 获得梯度后，如何更新
    learning_rate = 0.1
    # w1.data -= learning_rate * w1.grad.data 与下面式子等价
    w1.data.sub_(learning_rate * w1.grad.data)  # w1.data是获取保存weights的Tensor


    # 简化我们更新参数的操作
    # import torch.optim as optim
    # # create your optimizer
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # # in your training loop:
    # for i in range(steps):
    #     optimizer.zero_grad()  # zero the gradient buffers，必须要置零
    #     output = net(input)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()  # Does the update


    w1 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True)  # 需要求导的话，requires_grad=True属性是必须的。
    w2 = Variable(torch.Tensor([1.0, 2.0, 3.0]), requires_grad=True)

    z = w1 * w2 + w1  # 第二次BP出现问题就在这，不知道第一次BP之后销毁了啥。
    res = torch.mean(z)
    res.backward(retain_variables=True)  # 第一次求导没问题
    res.backward()  # 第二次BP会报错,但使用 retain_variables=True，就好了。
    # Trying to backward through the graph second time, but the buffers have already been
    # freed. Please specify retain_variables=True when calling backward for the first time

    #只使用部分 Variable 求出来的 loss对于原Variable求导得到的梯度是什么样的
    #from torch.autograd import Variable
    import torch.cuda as cuda
    w1 = Variable(cuda.FloatTensor(2, 3), requires_grad=True)
    res = torch.mean(w1[1])     # 只用了variable的第二行参数
    res.backward()
    print(w1.grad)

    v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
    h = v.register_hook(lambda grad: grad * 10)  # double the gradient
    v.backward(torch.Tensor([1, 1, 1]))
    # 先计算原始梯度，再进hook，获得一个新梯度。
    print(v.grad.data)
    h.remove()  # removes the hook