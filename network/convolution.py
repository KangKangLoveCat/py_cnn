#-*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import math
import numpy as np

def conv(X, W, b, stride = 1, padding = 0):
    n_filters, d_filter, kernel_size, _ = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - kernel_size + 2 * padding) / stride + 1
    w_out = (w_x - kernel_size + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col(X, kernel_size, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = (np.dot(W_col, X_col).T + b).T
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    return out

def im2col(x, kernel_size, padding=0, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, kernel_size, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel_size ** 2 * C, -1)
    return cols

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

def col2im(x, img_shape, kernel_size, padding = 0, stride = 1):
    x_row_num, x_col_num = x.shape
    channels, img_height, img_width = img_shape
    x_width = img_width - kernel_size + padding + 1
    x_height = img_height - kernel_size + padding + 1
    assert channels * kernel_size ** 2 == x_row_num
    assert x_width * x_height == x_col_num
    x_reshape = x.T.reshape(x_height, x_width, channels, kernel_size, kernel_size)
    output_padded = np.zeros((channels, img_height + 2 * padding, img_width + 2 * padding))
    for i in range(x_height):
        for j in range(x_width):
            output_padded[:, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size] = \
                    output_padded[:, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size] + \
                    x_reshape[i, j, ...]
    return output_padded[:, padding : img_height + padding, padding : img_width + padding]

class Convolution:

    def __init__(self, layer, kernel_size = 1, num_output = 1, padding = 0):
        self.upper_layer = layer
        self.num = layer.num
        self.num_input = layer.num_output
        self.input_h = layer.output_h
        self.input_w = layer.output_w
        self.output_h = self.input_h + 2 * padding - kernel_size + 1
        self.output_w = self.input_w + 2 * padding - kernel_size + 1
        self.num_output = num_output
        self.kernel_size = kernel_size
        self.padding = padding
        scale = math.sqrt(3. / (self.num_input * kernel_size**2))
        self.weight = np.random.rand(num_output, self.num_input, kernel_size, kernel_size)
        self.weight = (self.weight - 0.5) * 2 * scale
        self.weight_diff_his = np.zeros(self.weight.shape)
        self.bias = np.zeros((num_output))
        self.bias_diff_his = np.zeros(self.bias.shape)

    def forward(self):
        self.input_data = self.upper_layer.forward()
        self.num = self.upper_layer.num
        self.output_data = conv(self.input_data, self.weight, self.bias, padding = self.padding)
        return self.output_data

    def backward(self, diff):
        self.diff = np.zeros(self.input_data.shape)
        weight_diff = np.zeros(self.weight.shape)
        weight_diff = weight_diff.reshape(weight_diff.shape[0], -1)
        bias_diff = np.zeros((self.num_output))
        weight_reshape_T = self.weight.reshape(self.weight.shape[0], -1).T
        for i in range(self.num):
            input_data_col = im2col(self.input_data[[i]], self.kernel_size, self.padding)
            weight_diff = weight_diff + diff[i].reshape(diff[i].shape[0], -1).dot(input_data_col.T)
            bias_diff = bias_diff + np.sum(diff[i].reshape(diff[i].shape[0], -1), 1)
            tmp_diff = weight_reshape_T.dot(diff[i].reshape(diff[i].shape[0], -1))
            self.diff[i, ...] = col2im(tmp_diff, self.input_data.shape[1:], self.kernel_size, padding = self.padding)
        self.weight_diff_his = 0.9 * self.weight_diff_his + weight_diff.reshape(self.weight.shape)
        self.weight = self.weight * 0.9995 - self.weight_diff_his
        self.bias_diff_his = 0.9 * self.bias_diff_his + 2 * bias_diff
        self.bias = self.bias * 0.9995 - self.bias_diff_his
        self.upper_layer.backward(self.diff)

    def get_output(self):
        return self.output_data
