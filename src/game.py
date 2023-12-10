import copy
import random
import numpy as np
from constants import *
class board:
    def __init__(self,board, playerturn):
        self.board = board
        self.next_move = None
        self.won_games = np.zeros(9,dtype=np.int8)
        self.log = []
        self.last_move = None
        self.playerturn = playerturn
        self.game_is_over = False

    def is_valid_move(self,position):
        if (self.board[position[0]][position[1]] == 0) and (self.won_games[position[0]] == 0):
            if self.next_move == None :  
                return True
            elif self.next_move == position[0]:
                return True
        return False
    
    def execute(self,position):
        self.board[position[0]][position[1]] = self.playerturn
        if game_is_win(self.board[position[0]])[0]:
            self.won_games[position[0]] = self.playerturn
            self.game_is_over = check_if_game_over(self.won_games)
        elif game_is_tie(self.board[position[0]])[0]:
            self.won_games[position[0]] = 2
            self.game_is_over = check_if_game_over(self.won_games)
        if self.won_games[position[1]] != 0:
            self.next_move = None
        else:
            self.next_move = position[1]
        self.log.append((self.playerturn,position,self.next_move))
        self.playerturn = flip_player(self.playerturn)
        return
        
    def undo_move(self):
        position = self.log[-1][1]
        self.board[position[0]][position[1]] = 0
        if len(self.log)!= 1:
            self.next_move = self.log[-2][2]
            self.last_move = self.log[-2][1]
        else:
            self.next_move = None
            self.last_move = None
        if not check_if_game_over(self.board[position[0]]):
            self.won_games[position[0]] = 0
        self.log.pop(-1)
        self.playerturn = flip_player(self.playerturn)
        return
    
    def possible_moves(self):
        if self.next_move == None:
            possible_moves = np.array([],dtype=np.int8)
            for i in range(9):
                if self.won_games[i] == 0:
                    possible_moves = np.concatenate((possible_moves, np.where(self.board[i] == 0)[0] +9*i))
        else:
            possible_moves = np.where(self.board[self.next_move] == 0)[0] + self.next_move * 9
        actions = np.array([[j//9, j%9] for j in possible_moves])
        return actions

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

    
def flip_player(playerturn):
    if playerturn == 1: return -1
    elif playerturn == -1: return 1
    elif playerturn=="X": return "O"
    elif playerturn=="O": return "X"
    else: print("WTF")