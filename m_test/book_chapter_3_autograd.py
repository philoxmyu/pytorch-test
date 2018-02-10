#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-30 下午8:14
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com


from __future__ import print_function
import torch as t
from torch.autograd import Variable as V


def f(x):
    y = x ** 2 * t.exp(x)
    return y


def grad(x):
    dx = 2*x*t.exp(x) + x ** 2 * t.exp(x)
    return dx

if __name__ == "__main__":
    a = V(t.ones(3, 4), requires_grad = True)
    print(a)

    b = V(t.zeros(3, 4))
    print(b)
    c = a.add(b)
    print(c)

    d = c.sum()
    print("d:", d)
    d.backward()

    # 前者在取data后变为tensor，而后从tensor计算sum得到float
    # 后者计算sum后仍然是Variable
    print(c.data.sum(), c.sum())
    print("a grad", a.grad)

    # 此处虽然没有指定c需要求导，但c依赖于a，而a需要求导，
    # 因此c的requires_grad属性会自动设为True
    print(a.requires_grad, b.requires_grad, c.requires_grad)
    # 由用户创建的variable属于叶子节点，对应的grad_fn是None
    print(a.is_leaf, b.is_leaf, c.is_leaf)

    # c.grad是None, 因c不是叶子节点，它的梯度是用来计算a的梯度
    # 所以虽然c.requires_grad = True,但其梯度计算完之后即被释放
    print(c.grad is None)

    x = V(t.randn(3, 4), requires_grad=True)
    y = f(x)
    print("y:", y)

    y.backward(t.ones(y.size()))  # grad_variables形状与y一致
    print("x grad:", x.grad)
    print("x grad self:", grad(x))

    print("========== computer graph ===========")

    x = V(t.ones(1))
    b = V(t.rand(1), requires_grad=True)
    w = V(t.rand(1), requires_grad=True)
    y = w * x  # 等价于y=w.mul(x)
    z = y + b  # 等价于z=y.add(b)

    # grad_fn可以查看这个variable的反向传播函数，
    # z是add函数的输出，所以它的反向传播函数是AddBackward
    print("z: grad_fn", z.grad_fn) # grad_fn <AddBackward1 object at 0x7fcec023b410>

    # next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
    # 第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward
    # 第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有
    print("z grad_fn->next", z.grad_fn.next_functions) # ((<MulBackward1 object at 0x7f2df5aa55d0>, 0L), (<AccumulateGrad object at 0x7f2df5aa5690>, 0L))
    # variable的grad_fn对应着和图中的function相对应
    print(z.grad_fn.next_functions[0][0] == y.grad_fn) # True
    # 第一个是w，叶子节点，需要求导，梯度是累加的
    # 第二个是x，叶子节点，不需要求导，所以为None
    print("y: grad", y.grad_fn.next_functions) # ((<AccumulateGrad object at 0x7f294453f5d0>, 0L), (None, 0L))

    # 叶子节点的grad_fn是None
    print(w.grad_fn, x.grad_fn)


    # 计算w的梯度的时候，需要用到x的数值( ∂y∂w=x∂y∂w=x )，这些数值在前向过程中会保存成buffer，在计算完梯度之后会自动清空。
    # 为了能够多次反向传播需要指定retain_graph来保留这些buffer。
    # print("y.grad_fn.save:", y.grad_fn.saved_variables)

    # 使用retain_graph来保存buffer
    z.backward(retain_graph=True)
    print("first1:", w.grad)
    # 多次反向传播，梯度累加，这也就是w中AccumulateGrad标识的含义
    z.backward()
    print("first2:", w.grad)


    # PyTorch使用的是动态图，它的计算图在每次前向传播时都是从头开始构建，所以它能够使用Python控制语句
    # （如for、if等）根据需求创建计算图。这点在自然语言处理领域中很有用，它意味着你不需要事先构建所有可能用到的图的路径，
    # 图在运行时才构建。