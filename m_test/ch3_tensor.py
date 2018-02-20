#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-1-22 下午3:55
# @Author  : philoxmyu
# @Contact : philoxmyu@gmail.com

from __future__ import print_function
import torch as t

'''
从接口的角度来讲，对tensor的操作可分为两类：

1. `torch.function`，如`torch.save`等。
2. 另一类是`tensor.function`，如`tensor.view`等。

为方便使用，对tensor的大部分操作同时支持这两类接口，在本书中不做具体区分，如`torch.sum (torch.sum(a, b))`与`tensor.sum (a.sum(b))`功能等价。

而从存储的角度来讲，对tensor的操作又可分为两类：

1. 不会修改自身的数据，如 `a.add(b)`， 加法的结果会返回一个新的tensor。
2. 会修改自身的数据，如 `a.add_(b)`， 加法的结果仍存储在a中，a被修改了。

函数名以`_`结尾的都是inplace方式, 即会修改调用者自己的数据，在实际应用中需加以区分。
'''



if __name__ == "__main__":
    print("======== create tensor ========")
    a = t.Tensor(2,3)
    print(a)  # 数值取决于内存空间的状态

    # 用list的数据创建tensor
    b = t.Tensor([[1, 2, 3], [4, 5, 6]])
    print(b)
    print(b.tolist())

    # `tensor.size()`返回`torch.Size`对象，它是tuple的子类，但其使用方式与tuple略有区别
    b_size = b.size()
    print("b size type:", b_size, type(b_size))
    print("b number:", b.numel(), b.nelement()) # b中元素总个数，2*3，等价于b.nelement()

    # 创建一个和b形状一样的tensor
    c = t.Tensor(b_size)
    # 创建一个元素为2和3的tensor
    d = t.Tensor((2, 3))
    print("c:, d:", c, d)

    print("ones: zero:", t.ones(2, 3), t.zeros(2,3))
    print("arange: linspace", t.arange(1,6,2), t.linspace(1,10,3))
    print("randn: perm", t.randn(2,3), t.randperm(5)) # randperm:长度为5的随机排列

    # 通过`tensor.view`方法可以调整tensor的形状，但必须保证调整前后元素总数一致。
    # `view`不会修改自身的数据，返回的新tensor与源tensor共享内存，也即更改其中的一个，另外一个也会跟着改变。
    # 在实际应用中可能经常需要添加或减少某一维度，这时候`squeeze`和`unsqueeze`两个函数就派上用场了。
    a = t.arange(0, 6)
    print("a view:", a.view(2, 3))
    b = a.view(-1, 3)   # 当某一维为-1的时候，会自动计算它的大小
    print("b view:", b)

    c = b.unsqueeze(1)
    print(c, c.size())  # 注意形状，在第1维（下标从0开始）上增加“１”
    print("c[0]:", c[0])

    # `resize`是另一种可用来调整`size`的方法，但与`view`不同，它可以修改tensor的大小。
    # 如果新大小超过了原大小，会自动分配新的内存空间，而如果新大小小于原大小，则之前的数据依旧会被保存，看一个例子。
    b.resize_(1, 3)
    print("b:", b)

    b.resize_(3, 3)  # 旧的数据依旧保存着，多出的大小会分配新空间
    print("b2", b)

    print("============ index ============")
    a = t.randn(3, 4)
    b = a > 1 # [torch.ByteTensor of size 3x4]
    print("a:", a)
    print("b:", b)
    print(a[a>1]) # 等价于a.masked_select(a>1)
    print(a[t.LongTensor([0, 1])])  # 第0行和第1行

    a = t.arange(0, 16).view(4, 4)
    print("a:", a)
    # 选取对角线的元素
    index = t.LongTensor([[0, 1, 2, 3]])
    print("index:", index)
    print(a.gather(0, index)) #选取反对角线上的元素，注意与上面的不同

    # 选取两个对角线上的元素
    index = t.LongTensor([[0, 1, 2, 3], [3, 2, 1, 0]]).t()
    print("index:", index)
    b = a.gather(1, index)
    print("b:", b)

    # 把两个对角线元素放回去到指定位置
    c = t.zeros(4, 4)
    print("c:", c)
    print("c scatter", c.scatter_(1, index, b))


    print("=========== scatter ============")
    # 设置默认tensor，注意参数是字符串
    # t.set_default_tensor_type('torch.IntTensor')
    a = t.Tensor(2, 3)
    print(a)

    # 把a转成FloatTensor，等价于b=a.type(t.FloatTensor)
    b = a.float()
    print(b)

    print("=========== 归并操作 ==========")
    b = t.ones(2, 3)
    c = b.sum(dim=0, keepdim=True) # size中是否有"1"，取决于参数keepdim，keepdim=True会保留维度1
    print (b, c, c.size())

    b = a.t()
    print(b.is_contiguous())  # 矩阵的转置会导致存储空间不连续，需调用它的.contiguous方法将其转为连续。
    b.contiguous()

    print("========== tensor and numpy ===========")
    # Tensor和Numpy数组之间具有很高的相似性，彼此之间的互操作也非常简单高效。需要注意的是，Numpy和Tensor共享内存。
    # 由于Numpy历史悠久，支持丰富的操作，所以当遇到Tensor不支持的操作时，可先转成Numpy数组，处理后再转回tensor，其转换开销很小。
    # Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.
    import numpy as np
    a = np.ones([2, 3])
    print(a)

    b = t.from_numpy(a)
    print(b)

    c = t.Tensor(a)
    print("c:", c)
    print(c.numpy())

    print("========= broadcast ===========")
    # 广播法则(broadcast)是科学运算中经常使用的一个技巧，它在快速执行向量化的同时不会占用额外的内存/显存。 Numpy的广播法则定义如下：
    # 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
    # 两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算
    # 当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状
    # PyTorch当前已经支持了自动广播法则，但是笔者还是建议读者通过以下两个函数的组合手动实现广播法则，这样更直观，更不易出错：
    #
    # unsqueeze或者view：为数据某一维的形状补1，实现法则1
    # expand或者expand_as，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间。
    # 注意，repeat实现与expand相类似的功能，但是repeat会把相同数据复制多份，因此会占用额外的空间。

    a = t.ones(3, 2)
    b = t.zeros(2, 3, 1)
    print("a:", a, "b:", b)
    # # 自动广播法则
    # 第一步：a是2维,b是3维，所以先在较小的a前面补1 ，
    #               即：a.unsqueeze(0)，a的形状变成（1，3，2），b的形状是（2，3，1）,
    # 第二步:   a和b在第一维和第三维形状不一样，其中一个为1 ，
    # 可以利用广播法则扩展，两个形状都变成了（2，3，2）
    print("a+b:", a+b)  #自动广播

    ## 手动广播
    a1 = a.unsqueeze(0)  # change to 1*3*2
    print("a unsqueeze:", a1)
    print("a expand:", a1.expand(2,3,2))

    print(a.unsqueeze(0).expand(2,3,2) + b.expand(2,3,2))

    print("========== storage =========")
    # 内部结构
    # tensor的数据结构
    # 所示。tensor分为头信息区(Tensor)
    # 和存储区(
    #     Storage)，信息区主要保存着tensor的形状（size）、步长（stride）、数据类型（type）等信息，而真正的数据则保存成连续数组。由于数据动辄成千上万，因此信息区元素占用内存较少，主要内存占用则取决于tensor中元素的数目，也即存储区的大小。
    #
    # 一般来说一个tensor有着与之相对应的storage, storage是在data之上封装的接口，便于使用，而不同tensor的头信息一般不同，但却可能使用相同的数据
    a = t.arange(0, 6)
    print(a.storage())

    b = a.view(2,3)
    print(b.storage())
    # 一个对象的id值可以看作它在内存中的地址
    # storage的内存地址一样，即是同一个storage
    print(id(b.storage()) == id(a.storage()))
    a[1] = 100
    print("after a change:", b)

    c = a[2:]
    print("c:", c)
    print("data ptr:", c.data_ptr(), a.data_ptr())  # data ptr: 27886408 27886400 ; 2*4

    c[0] = -100  # c[0]的内存地址对应a[2]的内存地址
    print(c)

    d = t.Tensor(c.storage())
    print("d:", d)

    # 下面４个tensor共享storage
    print(id(a.storage()) == id(b.storage()) == id(c.storage()) == id(d.storage()))

    print("========== persist ==========")
    if t.cuda.is_available():
        a = a.cuda(0)  # 把a转为GPU1上的tensor,
        t.save(a, 'a.pth')

        # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
        b = t.load('a.pth')
        # # 加载为c, 存储于CPU
        # c = t.load('a.pth', map_location=lambda storage, loc: storage)
        # # 加载为d, 存储于GPU0上
        # d = t.load('a.pth', map_location={'cuda:1': 'cuda:0'})

    print("========= linear =========")
    # 设置随机数种子，保证在不同电脑上运行时下面的输出一致
    t.manual_seed(1000)
    def get_fake_data(batch_size=8):
        ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
        x = t.rand(batch_size, 1) * 20
        y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
        return x, y

    x, y = get_fake_data()
    print(x, y)
    print(x.squeeze(), y.squeeze())

    from matplotlib import pyplot as plt
    plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
    plt.show()

    # 随机初始化参数
    w = t.rand(1, 1)
    b = t.zeros(1, 1)
    lr = 0.001  # 学习率

    for ii in range(20000):
        x, y = get_fake_data()

        # forward：计算loss
        y_pred = x.mm(w) + b.expand_as(y)  # x@W等价于x.mm(w);for python3 only
        loss = 0.5 * (y_pred - y) ** 2  # 均方误差
        loss = loss.sum()

        # backward：手动计算梯度
        dloss = 1
        dy_pred = dloss * (y_pred - y)

        dw = x.t().mm(dy_pred)
        db = dy_pred.sum()

        # 更新参数
        w.sub_(lr * dw)
        b.sub_(lr * db)
