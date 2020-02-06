#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:38:22 2019

@author: turnerromey
"""
#import torch
#from torch.utils.data import Dataset
import numpy as np

class Chess_Data_Load():
    def __init__(self):
        my_data = np.load("processed_data_test.npz")
        self.X = my_data['arr_0']
        self.Y = my_data['arr_1']
        print(self.X.shape, self.Y.shape)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
    
    def get_x(self):
        return self.X
    
    def get_y(self):
        return self.Y
    
    def show_x_y(self):
        print(self.X)
        print("-------------")
        print(self.Y)
        return

