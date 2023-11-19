from game import *
    
def minmax(board, playerturn):
    if check_if_game_over(board):
        if game_is_win(board)[0]:
            return change_player_state(flip_player(playerturn)), None
        elif game_is_tie:
            return 0, None
    actions = []
    for index, value in enumerate(board):
        if value == '-':
            actions.append(index)
    if playerturn == 'X':
        value = -2
        for i in actions:
            playerflip = flip_player(playerturn)
            new_board = execute(playerturn, copy.copy(board),i)
            best = minmax(new_board,playerflip)[0]
            if best >= value:
                if best > value:
                    value = best
                    path = i
                    liste = [i]
                else:
                    liste.append(i)
    if playerturn == 'O':
        value = 2
        for i in actions:
            playerflip = flip_player(playerturn)
            new_board = execute(playerturn, copy.copy(board),i)
            best = minmax(new_board,playerflip)[0]
            if best <= value:
                if best < value:
                    value = best
                    liste =[i]
                else:
                    liste.append(i)
    return value , random.choice(liste)