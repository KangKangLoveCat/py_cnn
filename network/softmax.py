# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import numpy as np

class Softmax:
    
    def __init__(self, layer):
        self.upper_layer = layer

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        _, self.dim = self.input_data.shape
        t = np.exp(self.input_data - self.input_data.max(1).reshape(-1,1))
        self.softmax_data = t / t.sum(1).reshape(-1,1)
        self.softmax_data[self.softmax_data < 1e-30] = 1e-30
        return self.softmax_data

    def calc_loss(self, label):
        s = np.tile(np.arange(self.dim), self.num).reshape(self.input_data.shape)
        gt_index = s == label.reshape(-1, 1) 
        loss = 0 - np.average(np.log(self.softmax_data[gt_index]))
        self.diff = self.softmax_data.copy()
        self.diff[gt_index] = self.diff[gt_index] - 1.
        self.diff = self.diff / self.num
        return loss

    def backward(self, lr):
        self.upper_layer.backward(self.diff * lr)
