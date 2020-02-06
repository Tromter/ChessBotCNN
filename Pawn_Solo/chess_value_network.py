#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  27 01:25:35 2019

@author: turnerromey
"""
import numpy as np
import chess
from tensorChess import tensorChess

class chess_value_network(object):
    
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
            
    def value(self):
        #Bellman Equation
        #input of our value network is the bitboard representation
        #output:
        #black win = -1
        #white win = 1
        #draw = 0
        return 0
    
    def edges(self):
        return list(self.board.legal_moves)
    
    def process(self):
        #make sure current board state is valid
        assert self.board.is_valid()
        
        #create a uint8 list of the 64 board spaces, this will represent the 
        #state of the board for the NN
        board_state = np.zeros(64, np.uint8)
        
        #loop through each spot on board and check for pieces
        for i in range(64):
            #position of piece equals a check of each position
            piece_pos = self.board.piece_at(i)
            #if there is a piece
            if piece_pos is not None:
                #here we have a dictionary (because we only want numbers) 
                #that will turn a piece into a number based on the returned 
                #symbol, uppercase is white lowercase is black
                board_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[piece_pos.symbol()]
        
        #checking for castle availability on White side
        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert board_state[0] == 4
            board_state[0] = 7
                
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert board_state[7] == 4
            board_state[7] = 7
            
        #checking for castle availability on Black side
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert board_state[56] == 8+4
            board_state[56] = 8+7
                
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert board_state[63] == 8+4
            board_state[63] = 8+7
            
        #checking if en passant square is available
        if self.board.ep_square is not None:
            #asserting that we can still en passant (havent used it yet)
            assert board_state[self.board.ep_square] == 0
            board_state[self.board.ep_square] = 8
        
        
        
        board_state = board_state.reshape(8,8)
        
        #create a board state to hold positions and game data
        state = np.zeros((8,8,5), np.uint8)
        
        #translation of 0-3 columns to binary
        state[:, :, 0] = (board_state>>3)&1
        state[:, :, 1] = (board_state>>2)&1
        state[:, :, 2] = (board_state>>1)&1
        state[:, :, 3] = (board_state>>0)&1
        
        
        #5th column holds whose turn it is, 0 for black, 1 for white
        state[:, :, 4] = (self.board.turn*1.0)

        
        return state
        