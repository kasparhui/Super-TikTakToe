import copy
import pygame
import random
import game
from AI import all_AI
from draw_game import drawGameState
from constants import *
import numpy as np

def little_position(pos):
    if 18<= pos < 87: 
        return 0
    elif 87<= pos <= 156:
        return 1
    elif 156 < pos <= 225:
        return 2
    else: 
        return None

def play_game(AI_mode=False, AI_mode2 = False, player_turn='X'):
    pygame.init()
    pygame.display.set_caption('Super TikTakToe')
    screen = pygame.display.set_mode((WIDTH ,HEIGHT + 70 ))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 80)
    running = True
    board = game.board(np.zeros((9,9),dtype=np.int8), tic_tac_toe_mapping[player_turn])
    AI = all_AI(tic_tac_toe_mapping[player_turn])
    AI.board = board
    initial_playerturn = player_turn
    move_was_made = True
    while running:
        if not move_was_made:
            if (player_turn == initial_playerturn) and AI_mode:
                position = AI.make_move(board,'best_move',depth = 5, smart_depth= 25)
                board.execute(position)
                player_turn=short_mapping[player_turn]
                board.last_move = position
                move_was_made = True
                print('best_move AI', game.flip_player(board.playerturn), position)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                #return
                running = False
            if e.type == pygame.MOUSEBUTTONDOWN:
                if not move_was_made:
                    if ((initial_playerturn == player_turn) and (not AI_mode)) or ((initial_playerturn != player_turn) and (not AI_mode2)):
                        location = pygame.mouse.get_pos()
                        big_COL = location[1]//(WIDTH//3)
                        big_ROW = location[0]//(HEIGHT//3)
                        col_pos = location[1] % (WIDTH //3)
                        row_pos = location[0] % (HEIGHT //3)
                        col = little_position(col_pos)
                        row = little_position(row_pos) 
                        if (col == None) or (row ==None):
                            continue
                        position = (3*big_COL + big_ROW, 3*col + row)
                        if board.is_valid_move(position):
                            board.execute(position)
                            player_turn=short_mapping[player_turn]
                            board.last_move = position
                            move_was_made = True
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_u:
                    if player_turn == 'X' and AI_mode2:
                        if len(board.log) > 1:
                            board.undo_move()
                            board.undo_move()
                    else:
                        if board.log != []:
                            board.undo_move()
                            player_turn=short_mapping[player_turn]
        if not move_was_made:
            if (player_turn != initial_playerturn) and AI_mode2:
                position = AI.make_move(board,'naiv_MCTC', depth = 5, smart_depth= 25)
                board.execute(position)
                player_turn=short_mapping[player_turn]
                board.last_move = position
                move_was_made = True

        if board.game_is_over: 
            if game.game_is_win(board.won_games)[0]: 
                string = f'Spieler {short_mapping[player_turn]} hat gewonnen!'
            elif game.game_is_tie(board.won_games): 
                string = f'Unentschieden!'
            board.next_move = -1
            break

        drawGameState(screen, board)
        string = f'Spieler {player_turn} ist am Zug'
        text_surface = font.render(string, True, 'black')
        text_rect = text_surface.get_rect()
        text_rect.center = (WIDTH // 2, HEIGHT +25)
        move_was_made = False
        screen.blit(text_surface, text_rect)
        clock.tick(15)
        pygame.display.flip()
        
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    play_game(AI_mode,player_turn)
                    return
        winLine = game.game_is_win(board.won_games)[1]
        drawGameState(screen, board)
        if winLine == 1 or winLine == 2 or winLine == 3:
            pygame.draw.line(screen,'red',((winLine-1)*HEIGHT//3+HEIGHT//6,0),(15,(WIDTH,(winLine-1)*HEIGHT//3+HEIGHT//6)))
        elif winLine == 4 or winLine == 5 or winLine ==6:
            pygame.draw.line(screen,'red',(0,(winLine-4)*WIDTH//3+WIDTH//6),(WIDTH,(winLine-4)*WIDTH//3+WIDTH//6),15)
        elif winLine == 7:
            pygame.draw.line(screen,'red',(0,0),(WIDTH,HEIGHT),20)
        elif winLine == 8:
            pygame.draw.line(screen,'red',(0,HEIGHT),(WIDTH,0),20)
        else:
            pass
        text_surface = font.render(string, True, 'black')
        text_rect = text_surface.get_rect()
        text_rect.center = (WIDTH // 2, HEIGHT + 25)
        screen.blit(text_surface, text_rect)
        clock.tick(15)
        pygame.display.flip()


if __name__ == '__main__':
    play_game(True, True, 'X')