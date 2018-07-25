# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import numpy as np

class Softmax_loss:
    
    def __init__(self, layer1, layer2):
        self.upper_layer = layer1
        self.label_layer = layer2

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        _, self.dim = self.input_data.shape
        t = np.exp(self.input_data - self.input_data.max(1).reshape(-1,1))
        softmax_data = t / t.sum(1).reshape(-1,1)
        softmax_data[softmax_data < 1e-30] = 1e-30
        s = np.tile(np.arange(self.dim), self.num).reshape(self.input_data.shape)
        gt_index = s == self.label_layer.get_label().reshape(-1, 1) 
        self.loss = 0 - np.average(np.log(softmax_data[gt_index]))
        self.diff = softmax_data.copy()
        self.diff[gt_index] = self.diff[gt_index] - 1.
        self.diff = self.diff / self.num
        return self.loss

    def backward(self, lr):
        self.upper_layer.backward(self.diff * lr)

    def get_loss(self):
        return self.loss


