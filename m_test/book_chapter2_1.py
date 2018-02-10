#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-22 下午3:31
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

from __future__ import print_function
import torch as t

'''
定义网络时，需要继承`nn.Module`，并实现它的forward方法，把网络中具有可学习参数的层放在构造函数`__init__`中。
如果某一层(如ReLU)不具有可学习的参数，则既可以放在构造函数中，也可以不放，但建议不放在其中，而在forward中使用`nn.functional`代替。
'''

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    net = Net()
    print(net)

    # 只要在nn.Module的子类中定义了forward函数，backward函数就会自动被实现(利用
    # `Autograd`)。在
    # `forward`
    # 函数中可使用任何Variable支持的函数，还可以使用if、for循环、print、log等Python语法，写法和标准的Python写法一致。
    #
    # 网络的可学习参数通过
    # `net.parameters()`
    # 返回，`net.named_parameters`
    # 可同时返回可学习的参数及名称。
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())

    # forward函数的输入和输出都是Variable，只有Variable才具有自动求导功能，而Tensor是没有的，所以在输入时，需把Tensor封装成Variable。
    input = Variable(t.randn(1, 1, 32, 32))
    out = net(input)
    print("out size:", out.size(), out)

    net.zero_grad()  # 所有参数的梯度清零
    out.backward(Variable(t.ones(1, 10)))  # 反向传播

    output = net(input)
    target = Variable(t.arange(0, 10))
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print("loss:", loss)

    # 运行.backward，观察调用之前和调用之后的grad
    net.zero_grad()  # 把net中所有可学习参数的梯度清零
    print('反向传播之前 conv1.bias的梯度')
    print(net.conv1.bias.grad)
    loss.backward()
    print('反向传播之后 conv1.bias的梯度')
    print(net.conv1.bias.grad)

    ## 在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，例如随机梯度下降法(SGD)的更新策略如下：
    # weight = weight - learning_rate * gradient
    # 手动实现如下：
    # ```python
    # learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)  # inplace 减法
    # ```

    # 新建一个优化器，指定要调整的参数和学习率
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # 在训练过程中
    # 先梯度清零(与net.zero_grad()效果一样)
    optimizer.zero_grad()
    # 计算损失
    output = net(input)
    loss = criterion(output, target)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()