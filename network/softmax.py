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
        t = np.exp(self.input_data - self.input_data.max(1).reshape(-1,1))
        self.output_data = t / t.sum(1).reshape(-1,1)
        return self.output_data

    def get_output(self):
        return self.output_data
