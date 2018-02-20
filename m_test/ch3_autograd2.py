# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/12 下午2:04
"""

'''
extend autograd
1. 绝大多数的函数都可以使用autograd 实现反向求导
2. 自定义的Function 需要继承autograd.Function, 没有构造函数, forward 和 backward 函数都是静态方法
3. forward 函数的输入和输出都是tensor, backward 函数的输入和输出variable
4. backward 函数的输出和forward函数的输入意义对应, backward 函数的输入和forward 函数的输入是一一对应
5. backward 函数的grad_output 参数 即 t.autograd.backward中的grad_variables
6. 如果某一个输入不需要求导, 直接返回None
7. 反向传播可能需要利用前向传播的中间结果, 在前向传播过程中,需要保存中间结果, 否则前向传播结束后这些对象立即被释放
8. gradcheck 函数检测实现是否正确, 通过控制eps的大小可以控制容忍的误差

针对3的说明
1. backward函数的输入值和返回值是variable, 但在实际使用时autograd.Function会将输入的variable 提取为tensor, 并将提取的tensor
再封装成variable 返回.
2. 在backward 函数中要对variable进行操作,是为了能够计算梯度的梯度
3. 这种设计在pytorch 0.2中引入, 虽然能让autograd 具有高阶求导功能,但其也限制了Tensor的使用,因为autograd中反向传播的函数只能利用当前
已经有的Variable操作.
4. 为了更好灵活性, 也为了兼容旧版本代码, pytorch 还提供了另外一种扩展autograd的方法: 利用装饰器@once_differentiable


'''
import torch as t
from torch.autograd import Function, Variable


class MultiplyAdd(Function):
    @staticmethod
    def forward(ctx, w, x, b):
        print('type in forward', type(x))
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        print('type in backward', type(x), grad_output)
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1

        return grad_w, grad_x, grad_b

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + t.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print("saved variables:", ctx.saved_variables)
        output,  = ctx.saved_variables
        grad_x = output * (1 - output) * grad_output
        return grad_x

if __name__ == "__main__":
    x = Variable(t.ones(1))
    w = Variable(t.rand(1), requires_grad = True)
    b = Variable(t.rand(1), requires_grad = True)
    print("begin forward")
    z = MultiplyAdd.apply(w, x, b)
    print("begin backward")
    z.backward()
    # x 不需要求导, 中间过程还是会计算它的导数但随后被清空
    print("grad:", x.grad, w.grad, b.grad)


    print("==============")
    x = Variable(t.ones(1))
    w = Variable(t.rand(1), requires_grad = True)
    b = Variable(t.rand(1), requires_grad = True)
    print("begin forward")
    z = MultiplyAdd.apply(w, x, b)
    print("begin backward")
    print("z.grad_fn:", z.grad_fn)  # torch.autograd.function.MultiplyAddBackward
    print(z.grad_fn.apply(Variable(t.ones(1))))

    print("==============")
    x = Variable(t.Tensor([5]), requires_grad=True)
    y = x ** 2
    grad_x = t.autograd.grad(y, x, create_graph=True)
    print("grad x:", grad_x)
    grad_grad_x = t.autograd.grad(grad_x[0], x)
    print("grad x x:", grad_grad_x)

    print("=================")
    # 采用数值逼近的方式检验计算梯度的公式对不对
    test_input = Variable(t.randn(3,4), requires_grad=True)
    t.autograd.gradcheck(Sigmoid.apply, (test_input, ), eps=1e-3)
