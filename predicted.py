#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.datasets import mnist
import glob,os,cv2
import matplotlib.pyplot as plt

from Design.Models.Handwriting_Digits_Recognition import HDR
M = HDR()

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 手写图片目录
directory_name_1 = './data/right_re'
directory_name_2 = './data/wrong_re'
directory_name_3 = './data/za'

# MLP模型测试
M.model_test_mlp(directory_name_1,train_images,train_labels)
# CNN模型测试
M.model_test_cnn(directory_name_1,train_images,train_labels)

'''
# test for 8
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.axis('off')
    plt.imshow(train_images[i+9])
plt.show()
'''

'''
#test_8&6
plt.subplot(1,2,1)
plt.imshow(train_images[17])
plt.subplot(1,2,2)
plt.imshow(train_images[39])
plt.show()
'''

