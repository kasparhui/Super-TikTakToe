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

def play_game(AI_mode=False, player_turn='X'):
    pygame.init()
    pygame.display.set_caption('Super TikTakToe')
    screen = pygame.display.set_mode((WIDTH,HEIGHT + 70 ))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 80)
    running = True
    board = game.board(np.zeros((9,9),dtype=np.int8))
    AI = all_AI()
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                #return
                running = False
            if e.type == pygame.MOUSEBUTTONDOWN:
                if player_turn == 'X' or (player_turn == 'O' and (not AI_mode)):
                    location = pygame.mouse.get_pos()
                    big_COL = location[0]//(WIDTH//3)
                    big_ROW = location[1]//(HEIGHT//3)
                    col_pos = location[0] % (WIDTH //3)
                    row_pos = location[1] % (HEIGHT //3)
                    col = little_position(col_pos)
                    row = little_position(row_pos) 
                    if (col == None) or (row ==None):
                        continue
                    position = (3*big_COL + big_ROW, 3*col + row)
                    if board.is_valid_move(position):
                        board.execute(tic_tac_toe_mapping[player_turn],position)
                        player_turn=game.flip_player(player_turn)
                    #print(board.board,board.won_games,board.next_move)

        if  game.check_if_game_over(board.won_games): 
            if game.game_is_win(board.won_games)[0]: 
                string = f'Spieler {game.flip_player(player_turn)} hat gewonnen!'
            elif game.game_is_tie(board.won_games): 
                string = f'Unentschieden!'
            board.next_move = -1
            break
        if player_turn == 'O' and AI_mode:
            position = AI.best_move(board,-1)[1]
            board.execute(tic_tac_toe_mapping[player_turn],position)
            player_turn=game.flip_player(player_turn)
        drawGameState(screen, board)
        if  game.check_if_game_over(board.won_games): 
            if game.game_is_win(board.won_games)[0]: 
                string = f'Spieler {game.flip_player(player_turn)} hat gewonnen!'
            elif game.game_is_tie(board.won_games): 
                string = f'Unentschieden!'
            board.next_move = -1
            break
        string = f'Spieler {player_turn} ist am Zug'
        text_surface = font.render(string, True, 'black')
        text_rect = text_surface.get_rect()
        text_rect.center = (WIDTH // 2, HEIGHT +25)
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
            pygame.draw.line(screen,'red',(0,(winLine-1)*HEIGHT//3+HEIGHT//6),(WIDTH,(winLine-1)*HEIGHT//3+HEIGHT//6),15)
        elif winLine == 4 or winLine == 5 or winLine ==6:
            pygame.draw.line(screen,'red',((winLine-4)*WIDTH//3+WIDTH//6,0),((winLine-4)*WIDTH//3+WIDTH//6,WIDTH),15)
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
    play_game(True, 'X')