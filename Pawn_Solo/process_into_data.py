#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  27 12:50:44 2019

@author: turnerromey
"""

from chess_value_network import chess_value_network
import os
import chess.pgn
import chess
import numpy as np

def process_into_data():
    x = []
    y = []
    value = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    
    gn = 0
    for fn in os.listdir("KingBase2019-pgn"):
        pgn = open(os.path.join("KingBase2019-pgn", fn))
        test = True
        while test:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                print("***ERROR***")
                break
            result = game.headers['Result']
            if result not in value:
                continue
            my_val = value[result]
            board = game.board()
            
            print("parsing game %d, got %d examples" % (gn, len(x)))
            gn += 1
            
            
            for move in game.mainline_moves():
               board.push(move)
               state = chess_value_network(board)
               my_state = state.process()
               #print(value, my_state)
               x.append(my_state)
               y.append(my_val)
            if len(x) > 100000:
                return x,y
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == '__main__':
    x,y = process_into_data()
    np.savez("processed_data_test.npz", x, y)