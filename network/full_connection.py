# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import math
import numpy as np

class Full_connection:

    def __init__(self, layer, num_output = 1):
        self.num = layer.num
        self.upper_layer = layer
        self.num_input = layer.num_output
        self.num_output = num_output
        self.output_w = 1
        self.output_h = 1
        scale = math.sqrt(3. / self.num_input)
        self.weight = np.random.rand(self.num_input, self.num_output)
        self.weight = (self.weight - 0.5) * 2 * scale
        self.weight_diff_his = np.zeros(self.weight.shape)
        self.bias = np.zeros((num_output))
        self.bias_diff_his = np.zeros(self.bias.shape)

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        input_dims = len(self.input_data.shape)
        input_cols = self.input_data.reshape(self.input_data.shape[0], -1)
        self.output_data = input_cols.dot(self.weight) + self.bias
        return self.output_data

    def backward(self, diff):
        weight_diff = self.input_data.T.dot(diff)
        bias_diff = np.sum(diff, axis = 0)
        self.diff = diff.dot(self.weight.T)
        self.weight_diff_his = 0.9 * self.weight_diff_his + weight_diff
        self.weight = self.weight * 0.9995 - self.weight_diff_his
        self.bias_diff_his = 0.9 * self.bias_diff_his + 2 * bias_diff
        self.bias = self.bias * 0.9995 - self.bias_diff_his
        self.upper_layer.backward(self.diff)

    def get_output(self):
        return self.output_data
