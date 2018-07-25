# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import numpy as np

class Relu:
   
    def __init__(self, layer):
        self.upper_layer = layer
        self.num_output = layer.num_output
        self.num = layer.num
        self.output_w = layer.output_w
        self.output_h = layer.output_h

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        self.output_data = self.input_data.copy()
        self.output_data[self.output_data < 0] = 0
        return self.output_data

    def backward(self, diff):
        self.diff = diff.copy()
        self.diff[self.input_data < 0] == 0
        self.upper_layer.backward(self.diff)

    def get_output(self):
        return self.output_data
