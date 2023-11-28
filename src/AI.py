from game import *
import random
class all_AI:
    def __init__(self, player):
        self.player = player

    def make_move(self, board, mode):
        if mode == 'random_move':
            return self.random_move(board)
        elif mode == 'best_move':
            return self.best_move(board)
        else:
            print('Kein gültiger move')
        
    def best_move(self,board):
        possible_moves = self.possible_moves(board)
        values = np.zeros_like(possible_moves)
        return self.random_move(board)

    def random_move(self,board):
        position = np.random.choice(self.possible_moves(board))
        return (position//9, position%9)
    
    def possible_moves(self,board):
        if board.next_move == None:
            possible_moves = np.array([],dtype=np.int8)
            for i in range(9):
                if board.won_games[i] == 0:
                    possible_moves = np.concatenate((possible_moves, np.where(board.board[i] == 0)[0] +9*i))
        else:
            possible_moves = np.where(board.board[board.next_move] == 0)[0] + board.next_move * 9
        return possible_moves

        


