# -*- coding:utf-8 -*-
#@Time : 2021/2/19 11:51
#@Author: lyc
#@File : kNNtest.py
import kNN
group, labels = kNN.createDataset()
key = kNN.classify0([0, 1.0], group, labels, 3)
print(key)