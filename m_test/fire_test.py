# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/18 下午3:43
"""

import fire
def add(x, y):
    return x + y

def mul(**kwargs):
    a = kwargs['a']
    b = kwargs['b']
    return a * b

if __name__ == '__main__':
    fire.Fire()