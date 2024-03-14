#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras import models, layers
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import numpy as np
import os, cv2
import glob


class HDR(object):
    ### MLP（全连接神经网络）模型
    def mlp(self, train_images, train_labels):
        ## 数据预处理
        train_images = train_images.reshape((60000, 28 * 28)).astype('float')
        train_labels = to_categorical(train_labels)
        ## 构建网络
        # 如果当前目录下存在训练好的模型就不训练
        if os.path.exists('./mlp_model.h5'):
            model = tf.keras.models.load_model('./mlp_model.h5')
        else:
            model = models.Sequential()
            model.add(layers.Flatten(input_shape=(28, 28)))
            model.add(layers.Dense(units=128, activation='relu', input_shape=(28 * 28,),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001))),
            model.add(layers.Dropout(0.01))
            model.add(layers.Dense(units=32, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
            model.add(layers.Dropout(0.01))
            model.add(layers.Dense(units=10, activation='softmax'))
            # 编译方法
            model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            # 训练二十轮后保存模型
            model.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2)
            print('mlp网络训练完毕')
            model.save('mlp_model.h5')
            print('训练的模型已保存')

    ### CNN（卷积神经网络）模型
    def cnn(self, train_images, train_labels):
        ## 数据预处理
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
        train_labels = to_categorical(train_labels)
        ## 构建网络
        # 如果当前目录下存在训练好的模型就不训练
        if os.path.exists('./cnn_model.h5'):
            model = tf.keras.models.load_model('./cnn_model.h5')
        else:
            # 构建参数已经设置好的LeNet卷积神经网络
            network = self.LeNet()
            # 编译方法
            network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            # 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
            # 通过十轮训练后保存训练模型
            network.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)
            print('cnn训练完毕')
            network.save('cnn_model.h5')
            print('训练的模型已保存')

    ### 系参数已经调节好的LeNet卷积神经网络
    def LeNet(self):
        network = models.Sequential()
        network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        network.add(layers.AveragePooling2D((2, 2)))
        network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        network.add(layers.AveragePooling2D((2, 2)))
        network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
        network.add(layers.Flatten())
        network.add(layers.Dense(84, activation='relu'))
        network.add(layers.Dense(10, activation='softmax'))
        return network

    def image_prepare(self, pic_name):
        # 读取图像，第二个参数是读取方式
        img = cv2.imread(pic_name, 1)
        # 使用全局阈值，降噪
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # 把opencv图像转化为PIL图像
        im = Image.fromarray(cv2.cvtColor(th1, cv2.COLOR_BGR2RGB))
        # 灰度化
        im = im.convert('L')
        # 为图片重新指定尺寸
        im = im.resize((28, 28), Image.ANTIALIAS)
        # plt.imshow(im)
        # plt.show()
        # 图像转换为list
        im_list = list(im.getdata())
        # 图像灰度反转
        # result = [(255 - x) * 1.0 / 255.0 for x in im_list]
        # 图片降维
        im_list = (np.expand_dims(im_list, 0))
        return im_list

    ### 读取指定目录下的图片
    def img_for_show(self, directory_name):
        Img = []
        for filename in os.listdir(r"./" + directory_name):
            img = cv2.imread(directory_name + "/" + filename)
            Img.append(img)
        return Img

    ### 用自己手写的数字图片测试生成的MLP模型
    def model_test_mlp(self, directory_name, train_images, train_labels):
        # 获取待处理的图片
        Img = self.img_for_show(directory_name)
        img = glob.glob(os.path.join(directory_name, '*'))
        for p, i in zip(img, range(9)):
            # 图片预处理
            test_my_img = self.image_prepare(p)
            # 获取模型
            self.mlp(train_images, train_labels)
            # 图片输入模型测试并得出结果
            model = tf.keras.models.load_model('./mlp_model.h5')
            my_result = model.predict(test_my_img)
            # 结果展示
            plt.subplot(3, 3, i + 1)
            plt.title('predict_digit:%i' % np.argmax(my_result[0]))
            plt.axis('off')
            plt.imshow(Img[i])
        plt.show()
        print("MLP模型预测自己手写的数字结果见figure1")

    def model_test_cnn(self, directory_name, train_images, train_labels):
        Img = self.img_for_show(directory_name)
        img = glob.glob(os.path.join(directory_name, '*'))
        for p, i in zip(img, range(9)):
            test_my_img = self.image_prepare(p)
            test_my_img = test_my_img.reshape((28, 28, 1)).astype('float') / 255
            self.cnn(train_images, train_labels)
            model = tf.keras.models.load_model('./cnn_model.h5')
            my_result = model.predict(test_my_img)
            plt.subplot(3, 3, i + 1)
            plt.title('predict_digit:%i' % np.argmax(my_result[0]))
            plt.axis('off')
            plt.imshow(Img[i])
        plt.show()
        print("CNN模型预测自己手写的数字结果见figure1")
