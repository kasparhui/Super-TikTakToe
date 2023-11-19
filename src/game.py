import copy
import pygame
import random
class board:
    def __init__(self,board):
        self.board = board
        self.next_move = None
        self.win_games = ['-' for _ in range(9)]

    def is_valid_move(self,position):
        if (self.board[position[0]][position[1]] == '-') and (self.win_games[position[0]] == '-'):
            if self.next_move == None :  
                return True
            elif self.next_move == position[0]:
                return True
        return False
    
    def print_board(self):
        
        print("---------")
        print(self.board[0]+" | " +self.board[1]+" | " +self.board[2])
        print(self.board[3]+" | " +self.board[4]+" | " +self.board[5])
        print(self.board[6]+" | " +self.board[7]+" | " +self.board[8])

    def execute(self,player_turn,position):
        self.board[position[0]][position[1]] = player_turn
        if game_is_win(self.board[position[0]])[0]:
            self.win_games[position[0]] = player_turn
        elif game_is_tie(self.board[position[0]])[0]:
            self.win_games[position[0]] = 'N'
        if self.win_games[position[1]] != '-':
            self.next_move = None
        else:
            self.next_move = position[1]
        return 

def check_rows(little_board):
    a = ('N' != little_board[0]== little_board[1]==little_board[2] !="-")
    b = ('N' != little_board[3]==little_board[4]==little_board[5]!="-") 
    c = ('N' != little_board[6]==little_board[7]==little_board[8]!="-")
    if a: return True, 4
    elif b: return True, 5
    elif c: return True, 6
    else: return False, -1

def check_column(little_board):
    a = ('N' != little_board[0]==little_board[3]==little_board[6]!="-")
    b = ('N' != little_board[1]==little_board[4]==little_board[7]!="-") 
    c = ('N' != little_board[2]==little_board[5]==little_board[8]!="-")
    if a: return True, 1
    elif b: return True, 2
    elif c: return True, 3

    else: return False, -1
def check_diagonal(little_board):
    a = ('N' != little_board[0]==little_board[4]==little_board[8]!="-") 
    b = ('N' != little_board[2]==little_board[4]==little_board[6]!="-")
    if a: return True, 7
    elif b: return True, 8
    else: return False, -1

def game_is_tie(little_board):
    for i in little_board:
        if i=="-": return False, -1
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