# -*- coding:utf8 -*-
"""
@author: yuxm
@contact: philoxmyu@gmail.com
@time: 18/2/13 上午8:38
"""
'''
1. __str__ __repr__


'''


class Magic1(object):
    '''
    str是针对于让人更好理解的字符串格式化，而repr是让机器更好理解的字符串格式化。
    '''
    def __init__(self, world):
        self.world = world

    def __str__(self):
        return 'world is %s str' % self.world

    def __repr__(self):
        return 'world is %s repr' % self.world


class Magic2(object):
    def __init__(self, song):
        self.song = song

    def __hash__(self):
        return 1241


class Magic3(object):
    '''
    1. __getattr__是一旦我们尝试访问一个并不存在的属性的时候就会调用，
    而如果这个属性存在则不会调用该方法。

    2.__setattr__是设置参数的时候会调用到的魔法方法，相当于设置参数前的一个钩子。
    每个设置属性的方法都绕不开这个魔法方法，只有拥有这个魔法方法的对象才可以设置属性。
    在使用这个方法的时候要特别注意到不要被循环调用了


    '''
    def __init__(self, world):
        self.world = world

    def __getattr__(self, item):
        return item

    def __setattr__(self, name, value):
        if name == 'value':
            object.__setattr__(self, name, value - 100)
        else:
            object.__setattr__(self, name, value)


if __name__ == "__main__":
    t = Magic1('world_big')
    print str(t)
    print repr(t)

    t1 = Magic2('popo')
    t2 = Magic2('yuxm')
    print t1, hash(t1)
    print t2, hash(t2)

    t3 = Magic3('world123')
    print t3.world4
    t4 = Magic3(123)
    t4.value = 200
    print t4.value