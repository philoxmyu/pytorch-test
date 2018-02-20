# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/12 下午9:04
"""

'''
nn.Module 深入分析


1    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

_parameter: 保存用户直接设置的parameter; self.param1 = nn.Parameter(t.randn(3,3)) 会被检测到, 在字典中加入一个key为param1,
value 为对应parameter的item; 而self.submodule = nn.Linear(3,4)中的parameter则不会存于此
_modules: 子module; self.submodule = nn.Linear(3,4)指定的子module会保存于此
_buffers: 缓存;如batchnorm 使用momentum机制, 每次前向传播需用到上一次前向传播的结果
_backward_hooks 与 _forward_hooks: 钩子技术, 用来提取中间变量, 类似variable的hook
training: BatchNorm与Dropout层在训练阶段和测试阶段中采取的策略不同, 通过判断training值决定前向传播策略


2. nn.Module 对象在构造函数中的行为看起来有些怪异, 想要真正掌握其原理, 需要看两个魔法方法
* __getattr__: getattr(obj, 'attr1') == obj.attr1, getattr无法找到的交给__getattr__处理, 如果这个对象没有实现
    __getattr__ 方法, 则抛出Attribute异常
* __setattr__: obj.name = value 会直接调用setattr(obj,'name', value), 如果obj对象实现了__setattr__方法, setattr会
直接调用obj.__setattr__('name', value)
*  nn.Module 实现了自定义的__setattr__ 函数, 当执行module.name = value 时, 会在__setattr__ 中判断value 是否为Parameter
或nn.Module对象, 如果是则将这些对象加到_parameter 和 _modules 两个字典中; 如果是其它类型的对象, 如Variable, list, dict等,
则调用默认的操作,保存在__dict__中

3. ModuleList and Sequential;
* 在以上的例子中, 都是将每一层的输出直接作为下一层的输入,这种网络称为前馈传播网络(Feedforward Neural Network);
* 对于此类网络, 如果每次都写复杂的forward 函数会有些麻烦;
*有两种简化方式: ModuleList and Sequential;
    - Sequential 是一个特殊的Module, 它包含几个子Module, 前向传播时会将输入一层接一层得传下去;
    - ModuleList 也是一个特殊的Module, 可以包含几个子module, 可以像list一样使用它, 但不能直接把输入传给ModuleList

4. nn.functional and nn.Module
* nn 中绝大多数layer在functional中都有一个与之相对应的函数
* nn.functional中的函数和nn.Module的主要区别在于, 用nn.Module 实现的layers 是一个特殊的类, 会自动提取可学习的参数;
而nn.functional中的函数更像是纯函数
* 如果模型有可学习参数, 最好用nn.Module, 否则既可以使用nn.functional, 也可以使用nn.Module, 两者在性能上没有太大的差异;

5. register_forward_hook 和 register_backward_hook 函数的功能类似于variable的register_hook,可以在module前向传播或反向传播
时注册钩子
* 每次前向传播结束后执行hook: hook(module, input, output) -> none; 类似还有 register_forward_pre_hook
* 反向传播 hook(module, grad_input, grad_output) -> Tensor or None
* hook 函数不应该修改输入和输出, 并且在使用后应及时删除, 以避免每次运行hook 增加运行负载;
* hook 函数主要用在获取某些中间结果的情景, 如中间某一层的输出或某一层的梯度;

6. state_dict and load_state_dict
* 在Pytorch中保存模型十分简单, 所有的Module 对象都具有state_dict()函数, 返回当前Module所有的状态数据
* 这些状态数据保存后, 下次使用模型时即可利用model.load_state_dict()函数将状态加载进来

7. nn and autograd
* autograd.Function 利用Tensor对autograd技术的扩展, 为autograd 实现来新的运算op, 不仅要实现前向传播还要手动实现反向传播
* nn.Module 利用了autograd技术, 对nn的功能进行扩展, 实现了深度学习中更多的层. 只需实现前向传播功能, autograd即会自动实现反向传播
* nn.functional 是 一些autograd 操作的集合, 是经过封装的函数
* 如果某一操作在autograd中尚未支持, 需要利用autograd.Function 手动实现对应的前向传播和反向传播

'''


import torch as t
from torch import nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.param1 = nn.Parameter(t.randn(3,3)) # self.register_parameter('param1', nn.Parameter(t.randn(3,3)))
        self.submodel1 = nn.Linear(3, 4)

    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel1(x)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.list = [nn.Linear(3,4), nn.ReLU()]
        self.module_list = nn.ModuleList([nn.Conv2d(3,3,3), nn.ReLU()])

    def forward(self, *input):
        pass


if __name__ == "__main__":
    net = Net()
    print(net)
    print("net._modules", net._modules)
    print("net._param", net._parameters)

    print("net.param1", net.param1)
    print("net.param1_2", net._parameters['param1']) # net.param1 == net._parameters['param1']

    for name, param in net.named_parameters():
        print(name, param.size())

    for name, submodel in net.named_modules(): # 查看所有的子module,包括当前module
        print(name, submodel)

    ##### nn.Sequential
    net1 = nn.Sequential()
    net1.add_module('conv', nn.Conv2d(3,3,3))
    net1.add_module('batchnorm', nn.BatchNorm2d(3))
    net1.add_module('activation_layer', nn.ReLU())
    print("net1-->", net1)

    net2 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
    )
    print("net2-->", net2)
    from collections import OrderedDict
    net3 = nn.Sequential(
        OrderedDict(
            [
                ('conv1', nn.Conv2d(3, 3, 3)),
                ('bn1', nn.BatchNorm2d(3)),
                ('relu1', nn.ReLU())
            ]
        )
    )
    print("net3-->", net3)

    # 可以根据名字或序号取出子module
    print("net1.conv->", net1.conv)
    print("net2.conv->", net2[0])
    print("net3.conv->", net3.conv1)


    #3. modulelist: 是Module的子类, 当在Module中使用它时, 能自动识别为子module
    modellist = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
    input = Variable(t.randn(1,3))
    for model in modellist:
        input = model(input)

    model_test = MyModule()  # list 中的子module 并不能被主module 识别;
    print("model test-->", model_test)
    for name, param in model_test.named_parameters():
        print(name, param.size())

    #### loss function
    score = Variable(t.randn(3,2))
    label = Variable(t.Tensor([1,0,1])).long()  # label 必须是LongTensor
    criterion = nn.CrossEntropyLoss()
    loss = criterion(score, label)
    print("loss:", loss)

    # 4
    input = Variable(t.randn(2, 3))
    model = nn.Linear(3,4)
    output1 = model(input)
    output2 = nn.functional.linear(input, model.weight, model.bias)
    print(output1 == output2)





    # 图像相关层
    # from PIL import Image
    # from torchvision.transforms import ToTensor, ToPILImage
    # to_tensor = ToTensor() # img -> tensor
    # to_pil = ToPILImage()
    # lena = Image.open('lena.png')
    #
    # # 输入是一个batch, batch_size = 1
    # lena_tensor = to_tensor(lena)
    # print("lena tensor:", lena_tensor.size()) # (3L, 490L, 490L)
    # unsqueeze_lena_tensor = to_tensor(lena).unsqueeze(0)[:, 0:1]
    # print("unsqueeze lena tensor:", unsqueeze_lena_tensor.size()) # (1L, 3L, 490L, 490L)
    #
    # # 锐化卷积核
    # kernel = t.ones(3,3)/-9
    # kernel[1][1] = 1
    # print("kernel:", kernel)
    # conv = nn.Conv2d(1, 1, (3,3), 1, bias=False)
    # conv.weight.data = kernel.view(1, 1, 3, 3)
    # out = conv(Variable(unsqueeze_lena_tensor))
    # print("out:", type(out), out.size())
    # to_pil(out.data.squeeze(0))

