# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/14 下午2:31
"""
'''
1. 训练神经网络需要用到很多工具, 其中最重要的三部分是数据, 可视化, 和 GPU 加速
2. 继承DataSet 需要实现__getitem__ 和 __len__
3. torchvision.transforms
* torchvision: 提供了很多视觉图像处理工具, 其中transforms 模块提供了对PIL image 对象和Tensor 对象
的常用操作
* Scale: 保持长宽比不变, 调整图片尺寸
* CenterCrop, RandomCrop, RandomSizedCrop: 裁剪图片
* Pad: 填充
* ToTensor: 将PIL.Image 对象转成Tensor, 会自动将[0,255] 归一化至[0,1]

4.torchvision.datasets.ImageFolder
* ImageFolder 假设所有的文件按文件夹保存, 每个文件夹下存储同一个类别的图片,文件夹名为类名

5. DataLoader
* torch.utils.data.Dataset 只负责数据的抽象, 一次调用__getitem__只返回一个样本
* 训练神经网络时, 是对一个batch的数据进行操作, 同时还需要对数据进行shuffle和并行加速
* dataloader 是一个可迭代的对象, 我们可以像使用迭代器一样使用它
* 在数据处理中, 有时会出现某个样本无法读取等问题, 例如图片损坏; 可以在Dataloader中实现自定义的collate_fn, 将空对象过滤掉

6. WeightedRandomSampler
* 它会根据每个样本的权重选取数据, 在样本比例不均衡的问题中,可用它进行重采样
* 每个样本的权重weights, 共选取的样本总数num_sumples

7. torchvision 主要包括以下三个部分
*  models: 提供深度学习中各种经典网络结构及预训练好的模型, 主要包括 vgg系列, ResNet系列, Inception系列
*  datasets: 提供常用的数据集加载, 设计上都是继承 torch.utils.data.Dataset: 主要包括MNIST, ImageNet, COCO等
*  transforms: 提供常用的数据预处理操作, 主要包括对Tensor及PIL Image对象的操作

8. 可视化 Tensorboard and visdom
* 在训练神经网络时, 我们希望能更直观地了解训练情况, 包括损失曲线, 输入图片, 输出图片, 卷积核的参数分布等信息
* pip install visdom; nohup python -m visdom visdom.server
* visdom
 - env: 不同环境的可视化结果相互隔离, 互不影响
 - pane: 窗格用于可视化图像,数值等
 - visdom 同时支持Pytorch的tensor 和 numpy 的ndarray 两种数据结构, 但不支持Python的 int, float 等类型
 因此每次传入时需要将数据转成ndarry 或 tensor
'''

import torch as t


from PIL import Image
import numpy as np
import os

from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import models, datasets
from torch import nn
import visdom

class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        # 所有图片的绝对路径, 这里不实际加载图片, 只是指定路径, 当调用__getitem__时才会真正读图片
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):  # obj[index] == obj.__getitem__[index]
        img_path = self.imgs[index]
        # dog: 1 cat:0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img = Image.open(img_path)
        array = np.array(pil_img)
        data = t.from_numpy(array)
        return data, label

    def __len__(self):
        return len(self.imgs)


class DogCat2(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        # 所有图片的绝对路径, 这里不实际加载图片, 只是指定路径, 当调用__getitem__时才会真正读图片
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):  # obj[index] == obj.__getitem__[index]
        img_path = self.imgs[index]
        # dog: 1 cat:0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)

class DogCat3(DogCat2):
    def __getitem__(self, index):
        try:
            # 调用父类的获取函数, 即DogCat2.__getitem__(self, index)
            return super(DogCat3, self).__getitem__(index)
        except:
            return None, None



def my_collate_fn(batch):
    '''
        batch 中每个元素形如(data, label)
    '''

    # 过滤为None的对象
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch)  # 用默认的方式拼接过滤后的batch数据

if __name__ == "__main__":
    dataset = DogCat('ch5_data/dogcat')
    img, label = dataset[0]
    # 返回样本的形状不一; 返回样本的数值较大, 未归一化至[-1, 1]
    for img, label in dataset:
        print(img.size(), img.float().mean(), label)

    transform = T.Compose(
        [
            T.Scale(224),      # 缩放图片(Image), 保持长宽比不变, 最短边为224
            T.CenterCrop(224), # 从图片中间裁剪224 * 224 的图片
            T.ToTensor(),      # 将图片转成Tensor, 并归一化至[0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    dataset = DogCat2('ch5_data/dogcat', transforms=transform)
    for img, label in dataset:
        print(img.size(), label)


    # ImageFolder
    dataset = ImageFolder('ch5_data/dogcat_2')
    print("dataset index:", dataset.class_to_idx)
    print("dateset imgs:", dataset.imgs)
    # print dataset[0][0].show()

    normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    transform = T.Compose(
        [
            T.RandomSizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ]
    )
    dataset = ImageFolder('ch5_data/dogcat_2', transform=transform)
    print dataset[0][0].size()
    to_img = T.ToPILImage()
    # to_img(dataset[0][0]*0.2 + 0.4).show()

    # dataloader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    dataiter = iter(dataloader)
    imgs, labels = next(dataiter)
    print("imgs size:", imgs.size())

    # for batch_datas, batch_labels in dataloader:
    #     print("batch datas:", batch_datas, batch_labels)



    # collate_fn
    dataset = DogCat3('ch5_data/dogcat_wrong', transforms=transform)
    print "==>", dataset[-1][1]
    dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1)
    for batch_datas, batch_labels in dataloader:
        print(batch_datas.size(), batch_labels.size())
        print(type(batch_datas), type(batch_labels))

    # weights
    # 权重越大的样本被选中的概率越大
    dataset = ImageFolder('ch5_data/dogcat_2', transform=transform)
    weights = [2 if label == 1 else 1 for data, label in dataset]
    print("weights:", weights)
    sampler = WeightedRandomSampler(weights=weights, num_samples=9, replacement=True)
    dataloader = DataLoader(dataset, batch_size=3, sampler=sampler)
    for datas, labels in dataloader:
        print("labels list:", labels.tolist())


    # torchvison
    # 加载预训练好的模型, 如果不存在会下载
    # 预训练好的模型保存在 ~/.torch/models下面
    # resnet34 = models.resnet34(pretrained=True, num_classes=1000)
    # # 修改最后的全连接层为10分类问题
    # resnet34.fc = nn.Linear(512, 10)
    #
    # # 指定数据集路径为data/, 如果数据集不存在则进行下载
    # # train=False 获取测试集
    # dataset = datasets.MNIST('data/', download=True, train=False, transform=transform)

    to_pil = T.ToPILImage()
    to_pil(t.randn(3, 64, 64))

    # 新建一个client
    vis = visdom.Visdom(env=u'test1')
    x = t.arange(1, 30, 0.01)
    y = t.sin(x)
    vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})
    for ii in range(0, 10):
        x = t.Tensor([ii])
        y = x







