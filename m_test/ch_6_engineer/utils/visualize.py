# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/18 下午2:41
"""

'''
1. 在项目中, 我们可能会用到一些helper 方法, 一些方法可以同一放在utils中, 需要使用时在引入
2. 本次实验中只会用到plot方法,用于统计损失信息
'''

import visdom
import time
import numpy as np

class Visualize(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
            修改visdom配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        '''
            self.plot('loss', 1.00)
        '''

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x==0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):

        self.vis.images(img_.cpu().numpy(),
                        win = name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def plot_many(self, d):
        '''
            一次plot多个
            @params d: dict(name, value) i.e. ('loss', 0.11)
        '''

        for k, v in d.items():
            self.plot(k, v)


    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)


    def log(self, info, win='log_text'):
        '''
            self.log({'loss': 1, 'lr': 0.0001})
        '''

        self.log_text += ('[{time}][{info}<br>]'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))

        self.vis.text(self.log_text, win)


    def __getattr__(self, name):
        return getattr(self.vis, name)




