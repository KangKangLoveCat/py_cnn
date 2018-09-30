# -*- coding: utf-8 -*-

  ##########################
  #                        #
  #   @Author: KangKang    #
  #                        #
  ##########################

import os
import cv2
import struct
import numpy as np

def read_images(bin_file_name):
    binfile = open(bin_file_name, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    img_num = head[1]
    img_width = head[2]
    img_height = head[3]
    bits_size = img_num * img_height * img_width
    raw_imgs = struct.unpack_from('>' + str(bits_size) + 'B', buffers, offset)
    binfile.close()
    imgs = np.reshape(raw_imgs, head[1:])
    return imgs

def read_labels(bind_file_name):
    binfile = open(bind_file_name, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    img_num = head[1]
    offset = struct.calcsize('>II')
    raw_labels = struct.unpack_from('>' + str(img_num) + 'B', buffers, offset)
    binfile.close()
    labels = np.reshape(raw_labels, [img_num, 1])
    return labels

class Mnist_data:

    TRAIN = 'TRAIN'
    TEST = 'TEST'

    def __init__(self, data_dir):
        self.mode = self.TRAIN
        self.train_epoch = 0
        self.test_eopch = 0
        self.train_num = 64
        self.test_num = 100
        self.num = self.train_num # batch_size
        self.num_output = 1 # channels
        self.train_images = read_images(os.path.join(data_dir, 'train-images-idx3-ubyte')) / 256.
        self.train_labels = read_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        self.test_images = read_images(os.path.join(data_dir, 't10k-images-idx3-ubyte')) / 256.
        self.test_labels = read_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        self.train_img_num, self.output_h, self.output_w = self.train_images.shape
        self.test_img_num, _, _ = self.test_images.shape
        self.train_cur_index = 0
        self.test_cur_index = 0

    def next_batch_train_data(self):
        if self.train_cur_index + self.num >= self.train_img_num:
            t1 = np.arange(self.train_cur_index, self.train_img_num)
            t2 = np.arange(0, self.train_cur_index + self.num - self.train_img_num)
            self.output_train_index = np.append(t1, t2)
            self.train_epoch = self.train_epoch + 1
            self.train_cur_index = self.train_cur_index + self.num - self.train_img_num
        else:
            self.output_train_index = np.arange(self.train_cur_index, self.train_cur_index + self.num)
            self.train_cur_index = self.train_cur_index + self.num


    def next_batch_test_data(self):
        if self.test_cur_index + self.num >= self.test_img_num:
            t1 = np.arange(self.test_cur_index, self.test_img_num)
            t2 = np.arange(0, self.test_cur_index + self.num - self.test_img_num)
            self.output_test_index = np.append(t1, t2)
            self.test_epoch = self.test_eopch = + 1
            self.test_cur_index = self.test_cur_index + self.num - self.test_img_num
        else:
            self.output_test_index = np.arange(self.test_cur_index, self.test_cur_index + self.num)
            self.test_cur_index = self.test_cur_index + self.num


    def forward(self):
        if self.mode == self.TRAIN:
            self.output_images = self.train_images[self.output_train_index].reshape(self.num, 1, self.output_h, self.output_w)
            self.output_labels = self.train_labels[self.output_train_index].reshape(-1)
        elif self.mode == self.TEST:
            self.output_images = self.test_images[self.output_test_index].reshape(self.num, 1, self.output_h, self.output_w)
            self.output_labels = self.test_labels[self.output_test_index].reshape(-1)
        else:
            return None
        return self.output_images

    def backward(self, diff):
        pass

    def get_data(self):
        return self.output_images

    def get_label(self):
        return self.output_labels

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == self.TRAIN:
            self.num = self.train_num
        elif self.mode == self.TEST:
            self.num = self.test_num

