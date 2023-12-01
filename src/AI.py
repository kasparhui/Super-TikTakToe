from game import *
import random
import math

class all_AI:
    def __init__(self, player):
        self.player = player

    def make_move(self, board, mode = 'random_move', depth = 3):
        if mode == 'random_move':
            return self.random_move(board)
        elif mode == 'best_move':
            return self.best_move(board, depth)
        else:
            print('Kein gültiger move')

    def straight_count(self, temp_array):
    

        if ((np.count_nonzero(temp_array == 1) > 0) and (np.count_nonzero(temp_array == -1) > 0)) or (np.count_nonzero(temp_array == 2)>0):
            return 0
        else:
            return( np.sum(temp_array) ** 3)
            


    def little_evaluation(self, little_board):
        state = little_board.sum() - (2* np.count_nonzero(little_board == 2))
        for i in range (3):
            state += self.straight_count(np.array([little_board[3*i],little_board[3*i+1],little_board[3*i+2]]))
            state += self.straight_count(np.array([little_board[i],little_board[i+3],little_board[i+6]]))
        state += self.straight_count(np.array([little_board[0], little_board[4],little_board[8]]))
        state += self.straight_count(np.array([little_board[2], little_board[4], little_board[6]]))
        
        return 0.75 * math.tanh(state/10)

    def get_evalution(self, board, show_all_states = False):
        values = np.copy(board.won_games).astype(np.float64)
        for i in (np.where(values ==0)[0]):
            values[i] = self.little_evaluation(board.board[i])
        if show_all_states:
            print(values, self.little_evaluation(values))
        return self.little_evaluation(values)


    def best_move(self, board, depth):
        position = self.minmax(board, self.player, depth)[1]
        return position

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

    def minmax(self, board, playerturn, depth):
        if check_if_game_over(board.won_games):
            if game_is_win(board.won_games)[0]:
                return flip_player(playerturn), None
            elif game_is_tie(board.won_games)[0]:
                return 0, None
        if depth ==1:
            return self.get_evalution(board), None
        moves = self.possible_moves(board)
        actions = [(j//9, j%9) for j in moves]
        if playerturn == 1:
            value = -2
            for i in actions:
                playerflip = flip_player(playerturn)
                new_board = copy.copy(board)
                new_board.execute(playerturn, i)
                best = self.minmax(new_board,playerflip, depth-1)[0]
                if best >= value:
                    if best > value:
                        value = best
                        path = i
                        liste = [i]
                    else:
                        liste.append(i)
                new_board.undo_move()
        if playerturn == -1:
            value = 2
            for i in actions:
                playerflip = flip_player(playerturn)
                new_board = copy.copy(board)
                new_board.execute(playerturn,i)
                best = self.minmax(new_board,playerflip, depth-1)[0]
                if best <= value:
                    if best < value:
                        value = best
                        liste =[i]
                    else:
                        liste.append(i)
                new_board.undo_move()
        return value , random.choice(liste)
        


