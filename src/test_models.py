import game
import AI
import os
counter = 0
iterations = 10
for test in range(iterations):
    print(test)
    board = game.Board(playerturn = -(test % 2 )*2 +1)
    AI1 = AI.all_AI(1, path=os.path.join('models', 'neural_net' ,f'first_model_weights_short.pth'), args = (64, 4))
    AI2 = AI.all_AI(2, args=(64, 4))
    while not board.game_is_over:
        if (board.playerturn == 1 and test % 2 == 0) or (board.playerturn == -1 and test % 2 == 1):
            position = AI1.make_move(board, mode = 'best_nn_move')
        else:
            position = AI2.make_move(board, mode = 'best_nn_move')
        board.execute(position)
    print(board.won_games)
    if game.game_is_win(board.won_games)[0]:
        won_player = ((-board.playerturn) + 1)/2
        if test % 2 == 1:
            won_player = 1- won_player
        print(f'AI{-won_player + 2} hat gewonnen')
        print(won_player)
        counter += won_player
    else:
        counter += 0.5
        print('Unentschieden')

print(f'{counter}/{iterations}')
