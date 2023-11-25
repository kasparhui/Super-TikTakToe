from game import *
import random
class all_AI:
    def __init__(self):
        pass

    def best_move(self,board,player_turn):
        if board.next_move == None:
            valid_boards = np.where(board.won_games == 0)[0].astype(np.int8)
        else: 
            valid_boards = np.array([board.next_move],dtype=np.int8)
        print(valid_boards,type(valid_boards))


