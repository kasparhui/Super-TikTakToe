from game import *
import random
class all_AI:
    def __init__(self):
        pass
    def game_state(self):
        pass
    def best_move(self,board,player_turn):
        self.all_possible_moves = []
        if board.next_move == None:
            self.valid_positions = np.where(board.won_games == 0)[0].astype(np.int8)
        else: 
            self.valid_positions = np.array([board.next_move],dtype=np.int8)
        self.valid_boards =  board.board[self.valid_positions]
        for valid_board in self.valid_boards:
            all_valid = np.where(valid_board == 0)[0].astype(np.int8)
            self.all_possible_moves.append(all_valid)
        return self.random_move(board,player_turn)

    def random_move(self,board,player_turn):
        choice = random.randrange(len(self.valid_positions))
        position = (self.valid_positions[choice], np.random.choice(self.all_possible_moves[choice]))
        return None, position

        


