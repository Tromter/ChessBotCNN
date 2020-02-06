#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:15:54 2019

@author: turnerromey
"""


import numpy as np
import tensorflow as tf
import chess
from tensorChess import tensorChess
from Chess_Data_load import Chess_Data_Load
from chess_value_network import chess_value_network

class move_eval(object):
    def __init__(self):
        self.model = tf.keras.models.load_model('save_model/my_model')
    
    def __call__(self, s):
        brd = s.process()[None]
        output = self.model(brd)
        return float(output.numpy()[0][0][0][0])
        
def explore_leaves(s, v):
    x = []
    for e in s.edges():
        s.board.push(e)
        x.append((v(s), e))
        s.board.pop()
    return x

def getKey(item):
    return item[0]
    

