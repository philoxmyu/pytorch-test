# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/12 下午8:15
"""

'''
1. autograd 可实现深度学习模型, 但抽象程度较低;
2. torch.nn 的核心数据结构是Module;它是一个抽象的概念, 既可以表示神经网络中的某个层(layer), 也
可以表示一个包含很多层的神经网络; 常见的做法是extend nn.module

3. 在构造函数__init__ 中必须自己定义可学习的参数, 并封装成Parameter, Parameter是一种特殊的Variable, requires_grad 默认是True
4. forward 函数实现前向传播过程, 其输入可以是一个或多个Variable, 对x的任何操作也必须是variable 支持的操作
5. 无须写反向传播函数, 因其前向传播都是对variable进行操作, nn.Module 能够利用autograd 自动实现反向传播
6. layer(input) == layer.__call__(input), 在__call__函数中,主要调用的是layer.forward(x), 另外还对钩子
做了一些处理,所以实际使用中应尽量使用layer(x)而不是使用layer.forward()
7. Module 中的可学习参数,可以通过named_parameters 或 parameters 返回迭代器, 前者会给每个parameter附上名字,使其更具有辨识度
8. Module 能够自动检测到自己的parameter, 并将其作为可学习参数. 除了parameter, Module 还包含子Module, 主Module能够递归查找
子Module的parameter
'''

import torch as t
from torch import nn
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()  # 等价于nn.Module.__init__(self)
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Perceptron, self).__init__()
        self.layer1 = Linear(in_features, hidden_features)  # 此处的Linear 是前面自定义的全连接层
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)



if __name__ == "__main__":
    layer = Linear(4, 3)
    input = Variable(t.randn(2, 4))
    output = layer(input)
    print("output:", output)

    for name, parameter in layer.named_parameters():
        print(name, parameter)


    print("======= perceptron ========")
    percepron = Perceptron(3, 4, 1)
    for name, param in percepron.named_parameters():
        print(name, param.size())




