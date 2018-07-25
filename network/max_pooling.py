# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import math
import numpy as np

def im2col(X, kernel_size = 1, stride = 1):
    num, channels, height, width = X.shape
    surplus_height = (height - kernel_size) % stride
    surplus_width = (width - kernel_size) % stride
    pad_h = (kernel_size - surplus_height) % kernel_size
    pad_w = (kernel_size - surplus_width) % kernel_size
    X = np.pad(X, ((0,0),(0,0),(0,pad_h),(0,pad_w)), mode='constant')
    k,i,j = get_im2col_indices(X.shape, kernel_size, stride = stride)
    X_col = X[:,k,i,j].reshape(num * channels, kernel_size**2, -1)
    X_col = X_col.transpose(0,2,1)
    return X_col.reshape(-1, kernel_size**2)

def get_im2col_indices(x_shape, kernel_size, padding=0, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - kernel_size) % stride == 0
    assert (W + 2 * padding - kernel_size) % stride == 0
    out_height = int((H + 2 * padding - kernel_size) / stride + 1)
    out_width = int((W + 2 * padding - kernel_size) / stride + 1)
    i0 = np.repeat(np.arange(kernel_size), kernel_size)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(kernel_size), kernel_size * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def col2ims(x, img_shape, kernel_size, stride):
    x_row_num, x_col_num = x.shape
    img_n, img_c, img_h, img_w = img_shape
    o_h = int(math.ceil((img_h - kernel_size + 0.) / stride)) + 1
    o_w = int(math.ceil((img_w - kernel_size + 0.) / stride)) + 1
    assert img_n * img_c * o_h * o_w == x_row_num
    assert kernel_size**2 == x_col_num
    surplus_h = (img_h - kernel_size) % stride
    surplus_w = (img_w - kernel_size) % stride
    pad_h = (kernel_size - surplus_h) % stride
    pad_w = (kernel_size - surplus_w) % stride
    output_padded = np.zeros((img_n, img_c, img_h + pad_h, img_w + pad_w))
    x_reshape = x.reshape(img_n, img_c, o_h, o_w, kernel_size, kernel_size)
    for n in range(img_n):
        for i in range(o_h):
            for j in range(o_w):
                output_padded[n, :, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size] = \
                        output_padded[n, :, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size] + \
                        x_reshape[n, :, i, j, ...]
    return output_padded[:, :, 0 : img_h + pad_h, 0 : img_w + pad_w]

class Max_pooling:

    def __init__(self, layer, kernel_size = 1, stride = 1):
        self.num = layer.num
        self.num_output = layer.num_output
        self.num_input = layer.num_output
        self.input_h = layer.output_h
        self.input_w = layer.output_w
        self.output_h = int(math.ceil((self.input_h - kernel_size + 0.) / stride)) + 1
        self.output_w = int(math.ceil((self.input_w - kernel_size + 0.) / stride)) + 1
        self.upper_layer = layer
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        input_col = im2col(self.input_data, self.kernel_size, self.stride)
        tmp_index = np.tile(np.arange(input_col.shape[1]),input_col.shape[0]).reshape(input_col.shape)
        self.max_index = tmp_index == input_col.argmax(1).reshape(-1,1)
        self.output_data = input_col[self.max_index].reshape(self.num, self.num_input, self.output_h, self.output_w)
        return self.output_data

    def backward(self, diff):
        diff_col = np.zeros((self.num * self.num_input * self.output_h * self.output_w, self.kernel_size**2))
        diff_col[self.max_index] = diff.reshape(-1)
        self.diff = col2ims(diff_col, self.input_data.shape, self.kernel_size, self.stride)
        self.upper_layer.backward(self.diff)

    def get_output(self):
        return self.output_data
