# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

class Flatten:

    def __init__(self, layer):
        self.upper_layer = layer
        self.num_input = layer.num_output
        self.input_w = layer.output_w
        self.input_h = layer.output_h
        self.output_w = 1
        self.output_h = 1
        self.num_output = self.num_input * self.input_h * self.input_w
        self.num = layer.num
    
    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        self.output_data = self.input_data.reshape(self.input_data.shape[0], -1)
        return self.output_data

    def backward(self, diff):
        self.diff = diff.reshape(self.input_data.shape)
        self.upper_layer.backward(self.diff)

    def get_output(self):
        return self.output_data
