# -*- coding:utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import numpy as np
from network import *
from mnist_data import Mnist_data

def calc_accuracy(pred, truth):
    n = np.size(truth)
    return np.sum(pred.argmax(1) == truth.reshape(-1)) / (n + 0.)

print("Reading Data...")
data = Mnist_data('data/mnist')
print('Initing Network...')
conv1 = Convolution(data, kernel_size = 5, num_output = 20)
pool1 = Max_pooling(conv1, kernel_size = 2, stride = 2)
conv2 = Convolution(pool1, kernel_size = 5, num_output = 50)
pool2 = Max_pooling(conv2, kernel_size = 2, stride = 2)
flat1 = Flatten(pool2)
fc1 = Full_connection(flat1, 500)
relu1 = Relu(fc1)
fc2 = Full_connection(relu1, 10)
softmax = Softmax(fc2)
softmax_loss = Softmax_loss(fc2, data)
avg_loss = -1
base_lr = 0.01
for i in range(10000):
    learning_rate = base_lr * (1 + 0.0001 * i) ** (-0.75)
    if i % 100 == 0:
        print("Testing...")
        data.set_model(data.TEST)
        acc_sum = 0
        for j in range(100):
            accuracy = calc_accuracy(softmax.forward(), data.get_label())
            acc_sum = acc_sum + accuracy
        acc_sum = acc_sum / 100
        print("Accuracy = %.4f" % (acc_sum))
    data.set_model(data.TRAIN)
    softmax_loss.forward()
    if avg_loss == -1:
        avg_loss = softmax_loss.get_loss()
    else:
        avg_loss = avg_loss * 0.9 + 0.1 * softmax_loss.get_loss()
    print("iter = %-5d\tAvg loss = %.4f\t learning rate = %f" % (i + 1, avg_loss, learning_rate))
    softmax_loss.backward(learning_rate)
