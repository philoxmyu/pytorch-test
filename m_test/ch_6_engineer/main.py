# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/18 下午1:19
"""
'''
1. 验证相对来说比较简单, 但要注意需要将模型置于验证模式(model.eval()), 验证完成后还需要将其置回为训练模式(model.train()),
这两句代码会影响BatchNorm 和 DropOut等层
2. 为了方便他人使用, 程序中还应该提供一个帮助函数, 用于说明函数是如何使用的. 程序的命令行接口中有众多的参数,如果手动用字符串表示不仅复杂,
后期修改config文件时还需要修改对应的帮助信息, 十分不方便; 这里使用inspect方法自动获取config的源码
3. fire 会将包含"-" 的命令行参数自动转成下画线"_", 也会将非数字的值转成字符串, 所以--train-data-root=data/train ==
--train_data_root='data/train'
i.e.:
    python main.py train
        --train_data_root=data/train/
        --load_model_path='checkpoints/resnet34_16:53:00.pth'
        --lr=0.005
        --batch_size=32
        --model='ResNet34'
        --max_epoch=20
'''
from config import opt
import os
import torch
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.visualize import Visualize
from torchnet import meter


def val(model, dataloader):
    '''
        计算模型在验证集上的准确率等信息
    '''
    # 把模型设置为验证模型
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label, volatile=True)
        if opt.use_gpu:
            val_intput = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.long())

    # 把模型恢复为训练模式
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def write_csv(results, file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def test(**kwargs):
    opt.parse(kwargs)
    # 模型
    model = getattr(models, opt.model).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 数据
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = torch.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = torch.nn.functional.softmax(score)[:,1].data.tolist() # 计算每个样本属于狗的概率
        batch_results = [(path_, probability_) for path_,probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results, opt.result_file)
    return results


def train(**kwargs):
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualize(opt.env)

    # step1: 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: 数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)


    #step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    #step4: 统计指标: 平滑处理后的损失, 还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        for ii, (data, label) in enumerate(train_dataloader):
            # 训练模型参数
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标及可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)
            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
    model.save()

    # 计算验证集上的指标及可视化
    val_cm, val_accuracy = val(model, val_dataloader)
    vis.plot('val_accuracy', val_accuracy)
    vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}".format(
        epoch=epoch,
        loss=loss_meter.value()[0],
        val_cm=str(val_cm.value()),
        train_cm=str(confusion_matrix.value()),
        lr=lr
    ))

    # 如果损失不再下降, 则降低学习率
    if loss_meter.value()[0] > previous_loss:
        lr = lr * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    previous_loss = loss_meter.value()[0]


def help():
    '''
        打印帮助信息: python main.py help
    '''
    print('''
        usage: python {0}  <function> [--args=value,]
        <function> := train | test | help
        example:
            python {0} train --env='env02189' --lr=0.1
            python {0} test --dataset='path/to/dataset/root/'
            pytyon {0} help
        aviable args.:'''.format(__file__))
    from inspect import getsource
    source = getsource(opt.__class__)
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()

