# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/18 下午2:13
"""

'''
    BasicModule 是对nn.Module的简易封装, 提供快速加载和保存模型的接口
'''
import torch
from torch import nn
import time
class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
            保存模型, 默认使用 模型名字+时间 作为文件名
        '''

        if name is None:
            prefix = 'checkpoints/' + self.model_name + "_"
            name = time.strftime(prefix, '%m%d_%H:%M:%S.pth')

        torch.save(self.state_dict(), name)
        return name
