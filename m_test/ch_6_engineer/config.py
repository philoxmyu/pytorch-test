# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/18 下午1:19
"""

'''
1. 在模型定义, 数据处理和训练过程中有很多变量, 这些变量应提供默认值, 并统一放在放置在配置文件中
'''
import warnings

class DefaultConfig(object):
    env = 'default'    # visdom 环境
    model = 'AlexNet'  # 使用的模型, 名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train'  # 训练集存放路径
    test_data_root = './data/test1'   # 测试集存放路径
    load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径,为None代表不加载

    batch_size = 128  # batch_size
    use_gpu = True    # use GPU or not
    num_workers = 4   # how many workes for loading data
    print_freq = 20   # print info every N batch

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95
    weight_decay = 1e-4


def parse(self, kwargs):
    '''
        根据字典kwargs 更新config 参数
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)

    # 打印配置信息
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

if __name__ == "__main__":
    '''
        在程序中可以如下使用配置参数
    '''
    import models
    from config import DefaultConfig
    opt = DefaultConfig()
    lr = opt.lr
    model = getattr(models, opt.model)
