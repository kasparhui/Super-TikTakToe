import copy
import random
import numpy as np
from constants import *
from torch import tensor, float32
 
class Board:
    def __init__(self, board = np.zeros((9,9), dtype=np.int8), playerturn = 1):
        self.board = copy.deepcopy(board)
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
        if type(position) == int:
            position = [position//9, position%9]
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
        self.game_is_over = False
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
    
    def possible_move_vector(self):
        legal_moves = np.zeros((9, 9), dtype=np.int8)
        if self.next_move == None:
            for i in range(9):
                if self.won_games[i] == 0:
                    legal_moves[i, np.where(self.board[i] == 0)[0]] = 1
        else:
            legal_moves[self.next_move, np.where(self.board[self.next_move] == 0)[0]] = 1
        return legal_moves
    
    def get_encoded_state(self):
        # Set a one on the board where the player can play a legal move
        possible_moves_board = np.zeros((9, 9))
        possible_moves = self.possible_moves()
        for x, y in possible_moves:
            possible_moves_board[x][y] = 1
        possible_moves_board = possible_moves_board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
        
        # Encode the board
        transposed_board = self.board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
        
        if self.playerturn == 1:
            encoded_state = np.stack((
                transposed_board == 1, 
                transposed_board == 0, 
                transposed_board == -1, 
                possible_moves_board
            )).astype(np.float32)
        else:
            encoded_state = np.stack((
                transposed_board == -1, 
                transposed_board == 0, 
                transposed_board == 1, 
                possible_moves_board
            )).astype(np.float32)

        encoded_state = tensor(encoded_state, dtype=float32)
        return encoded_state
    
    def change_perspective(self):
        if self.playerturn == 1:
            return
        
        self.board = -self.board
        self.won_games = -self.won_games
        self.won_games = np.where(self.won_games == -2 , 2 , self.won_games)
        self.playerturn = flip_player(self.playerturn)

def check_rows(little_board):
    first_row = (2 != little_board[0] == little_board[1] == little_board[2] != 0)
    second_row = (2 != little_board[3] == little_board[4] == little_board[5] != 0) 
    third_row = (2 != little_board[6] == little_board[7] == little_board[8] != 0)
    
    won = first_row or second_row or third_row
    row = 4 if first_row else 5 if second_row else 6 if third_row else -1
    return won, row

def check_column(little_board):
    first_column = (2 != little_board[0] == little_board[3] == little_board[6] != 0)
    second_column = (2 != little_board[1] == little_board[4] == little_board[7] != 0) 
    third_column = (2 != little_board[2] == little_board[5] == little_board[8] != 0)
    
    won = first_column or second_column or third_column
    column = 1 if first_column else 2 if second_column else 3 if third_column else -1
    return won, column

def check_diagonal(little_board):
    top_left_to_bottom_right = (2 != little_board[0]==little_board[4]==little_board[8]!=0) 
    top_right_to_bottom_left = (2 != little_board[2]==little_board[4]==little_board[6]!=0)
    
    won = top_left_to_bottom_right or top_right_to_bottom_left
    diagonal = 7 if top_left_to_bottom_right else 8 if top_right_to_bottom_left else -1
    return won, diagonal

def game_is_tie(little_board):
    for i in little_board:
        if i==0: return False, -1
    return True, 0

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
    return -playerturn