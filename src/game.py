import copy
import pygame
import random
class board:
    def __init__(self,board):
        self.board = board
        self.next_move = None

    def is_valid_move(self,position):
        if self.board[position[0]][position[1]] == '-':
            if self.next_move == None: 
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
        self.next_move = position[1]
        return 

    def check_rows(self):
        a = (self.board[0]== self.board[1]==self.board[2] !="-")
        b = (self.board[3]==self.board[4]==self.board[5]!="-") 
        c = (self.board[6]==self.board[7]==self.board[8]!="-")
        if a: return True, 4
        elif b: return True, 5
        elif c: return True, 6
        else: return False, -1


    def check_column(self):
        a = (self.board[0]==self.board[3]==self.board[6]!="-")
        b = (self.board[1]==self.board[4]==self.board[7]!="-") 
        c = (self.board[2]==self.board[5]==self.board[8]!="-")
        if a: return True, 1
        elif b: return True, 2
        elif c: return True, 3
        else: return False, -1

    def check_diagonal(self):
        a = (self.board[0]==self.board[4]==self.board[8]!="-") 
        b = (self.board[2]==self.board[4]==self.board[6]!="-")
        if a: return True, 7
        elif b: return True, 8
        else: return False, -1


    def game_is_tie(self):
        for i in self.board:
            if i=="-": return False, -1
        else: return True, 0


    def game_is_win(self):
        row_winner=self.check_rows()
        column_winner=self.check_column()
        diagonal_winner=self.check_diagonal()
        if row_winner[0]:
            return True, row_winner[1]
        elif column_winner[0]:
            return True, column_winner[1]
        elif diagonal_winner[0]:
            return True, diagonal_winner[1]
        else: return False, 0


    def check_if_game_over(self):
        if self.game_is_tie()[0]: return (True)
        is_win=self.game_is_win()[0]
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