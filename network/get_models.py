# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/27 19:28
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from network import DORN_nyu, DORN_kitti


def get_models(dataset='nyu', pretrained=True, freeze=True):
    if dataset == 'kitti':
        rgb_size = (385, 513)
    else:
        rgb_size = (257, 353)
    return DORN_nyu.DORN(pretrained=pretrained,freeze=freeze,output_size=rgb_size)
    # else:
    #     print('no model based on dataset-', )
    #     exit(-1)

