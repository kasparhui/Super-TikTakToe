import copy
import pygame
import random
import numpy as np
from constants import *
class board:
    def __init__(self,board):
        self.board = board
        self.next_move = None
        self.won_games = np.zeros(9,dtype=np.int8)
        self.log = []
    def is_valid_move(self,position):
        if (self.board[position[0]][position[1]] == 0) and (self.won_games[position[0]] == 0):
            if self.next_move == None :  
                return True
            elif self.next_move == position[0]:
                return True
        return False
    
    def execute(self,player_turn,position):
        self.board[position[0]][position[1]] = player_turn
        if game_is_win(self.board[position[0]])[0]:
            self.won_games[position[0]] = player_turn
        elif game_is_tie(self.board[position[0]])[0]:
            self.won_games[position[0]] = 2
        if self.won_games[position[1]] != 0:
            self.next_move = None
        else:
            self.next_move = position[1]
        self.log.append((player_turn,position,self.next_move))
        return 
    
    def undo_move(self):
        position = self.log[-1][1]
        self.board[position[0]][position[1]] = 0
        if len(self.log)!= 1:
            self.next_move = self.log[-2][2]
        else:
            self.next_move = None
        if not check_if_game_over(self.board[position[0]]):
            self.won_games[position[0]] = 0
        self.log.pop(-1)

def check_rows(little_board):
    a = (2 != little_board[0]== little_board[1]==little_board[2] !=0)
    b = (2 != little_board[3]==little_board[4]==little_board[5] !=0) 
    c = (2 != little_board[6]==little_board[7]==little_board[8] !=0)
    if a: return True, 4
    elif b: return True, 5
    elif c: return True, 6
    else: return False, -1

def check_column(little_board):
    a = (2 != little_board[0]==little_board[3]==little_board[6]!=0)
    b = (2 != little_board[1]==little_board[4]==little_board[7]!=0) 
    c = (2 != little_board[2]==little_board[5]==little_board[8]!=0)
    if a: return True, 1
    elif b: return True, 2
    elif c: return True, 3

    else: return False, -1
def check_diagonal(little_board):
    a = (2 != little_board[0]==little_board[4]==little_board[8]!=0) 
    b = (2 != little_board[2]==little_board[4]==little_board[6]!=0)
    if a: return True, 7
    elif b: return True, 8
    else: return False, -1

def game_is_tie(little_board):
    for i in little_board:
        if i==0: return False, -1
    else: return True, 0

def game_is_win(little_board):
    row_winner=check_rows(little_board)
    column_winner=check_column(little_board)
    diagonal_winner=check_diagonal(little_board)
    if row_winner[0]:
        return True, row_winner[1]
    elif column_winner[0]:
        return True, column_winner[1]
    elif diagonal_winner[0]:
        return True, diagonal_winner[1]
    else: return False, 0

def check_if_game_over(little_board):
    if game_is_tie(little_board)[0]: return (True)
    is_win=game_is_win(little_board)[0]
    if is_win: 
        return (True)
    else: return False

def change_player_state(playerturn):
    dic = {'O':-1,'X':1}
    return(dic[playerturn])

    
def flip_player(player_turn):
    if player_turn=="X": return "O"
    elif player_turn=="O": return "X"
    else: print("WTF")