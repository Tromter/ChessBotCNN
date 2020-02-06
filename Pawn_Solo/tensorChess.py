#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 1:32:14 2019

@author: turnerromey
"""
#import torch
#from torch.utils.data import Dataset

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from Chess_Data_load import Chess_Data_Load

class tensorChess(tf.keras.Model):
    
    def __init__(self):
        
        super(tensorChess, self).__init__()
        
        #(input = 5, output = 16, kernel_size = 3, padding = 1)
        self.a1 = layers.Conv2D(16, kernel_size=3, padding='SAME', data_format='channels_last')
        self.a2 = layers.Conv2D(16, kernel_size=3, padding='SAME', data_format='channels_last')
        self.a3 = layers.Conv2D(32, kernel_size=3, strides=2, padding='VALID', data_format='channels_last')
        
        self.b1 = layers.Conv2D(32, kernel_size=3, padding='SAME', data_format='channels_last')
        self.b2 = layers.Conv2D(32, kernel_size=3, padding='SAME', data_format='channels_last')
        self.b3 = layers.Conv2D(64, kernel_size=2, strides=2, padding='VALID', data_format='channels_last')
        
        self.c1 = layers.Conv2D(64, kernel_size=2, padding='SAME', data_format='channels_last')
        self.c2 = layers.Conv2D(64, kernel_size=2, padding='SAME', data_format='channels_last')
        self.c3 = layers.Conv2D(128, kernel_size=1, strides=2, padding='VALID', data_format='channels_last')
        
        self.d1 = layers.Conv2D(128, kernel_size=1, padding='VALID', data_format='channels_last')
        self.d2 = layers.Conv2D(128, kernel_size=1, padding='VALID', data_format='channels_last')
        self.d3 = layers.Conv2D(128, kernel_size=1, padding='VALID', data_format='channels_last')
        
        self.density = layers.Dense(1)
                
    def call(self, x):
        #initial 8x8
        x = tf.nn.relu(self.a1(tf.cast(x, tf.float32)))
        x = tf.nn.relu(self.a2(x))
        x = tf.nn.relu(self.a3(x))
        #x = tf.nn.max_pool2d(x, 2, 2, padding='VALID')
        
        #4x4
        x = tf.nn.relu(self.b1(x))
        x = tf.nn.relu(self.b2(x))
        x = tf.nn.relu(self.b3(x))
        #x = tf.nn.max_pool2d(x, 2, 2, padding='VALID')
        
        #2x2
        x = tf.nn.relu(self.c1(x))
        x = tf.nn.relu(self.c2(x))
        x = tf.nn.relu(self.c3(x))
        #x = tf.nn.max_pool2d(x, 2, 2, padding='VALID')
        
        #1x128
        x = tf.nn.relu(self.d1(x))
        x = tf.nn.relu(self.d2(x))
        x = tf.nn.relu(self.d3(x))
        
        tf.reshape(x, (-1, 128))
        
        x = self.density(x)
        #x = tf.contrib.slim.fully_connected(x, 1, activation_fn = None)
        
        #value output
        #return tf.nn.sigmoid(x)
        return tf.math.tanh(x)
    



