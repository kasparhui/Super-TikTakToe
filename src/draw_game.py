import pygame
from constants import *

def draw_big_grid(screen):
    for i in range(1,3):
        pygame.draw.rect(screen, 'black', pygame.Rect(i*WIDTH//3-10, 9, 20, WIDTH-18))
        pygame.draw.rect(screen, 'black', pygame.Rect(9, i*HEIGHT//3-10, HEIGHT-18, 20))

def draw_little_grid_and_next_moves(screen,board,big_row,big_col):
    if board.next_move == (big_row*3 + big_col):
        pygame.draw.rect(screen, (243,219,214), pygame.Rect(10+big_row*WIDTH//3,10+big_col*HEIGHT//3,WIDTH//3-20,HEIGHT//3-20))
    elif board.next_move == None and board.won_games[3*big_row+big_col] == 0:
        pygame.draw.rect(screen, (243,219,214), pygame.Rect(10+big_row*WIDTH//3,10+big_col*HEIGHT//3,WIDTH//3-20,HEIGHT//3-20))
    for i in range(1,3):
        pygame.draw.rect(screen, 'black', pygame.Rect(i*(WIDTH//3-36)//3+big_row*WIDTH//3+13, big_col*HEIGHT//3+18, 10, WIDTH//3-36))
        pygame.draw.rect(screen, 'black', pygame.Rect(big_row*WIDTH//3+18, i*(HEIGHT//3-36)//3 + big_col*HEIGHT//3+13, HEIGHT//3-36, 10))

def draw_little_state(screen,board,big_row,big_col,row,col):
    temp_board = board.board[big_row*3 + big_col]
    top_left = (big_row*WIDTH//3+row*((WIDTH//3-36)//3)+32,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+26)
    top_right = (big_row*WIDTH//3+row*((WIDTH//3-36)//3)+73,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+26)
    bottom_right = (big_row*WIDTH//3+row*((WIDTH//3-36)//3)+73,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+79)
    bottom_left = (big_row*WIDTH//3+row*((WIDTH//3-36)//3)+32,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+79)
    if temp_board[3*row + col] == -1:
        pygame.draw.circle(screen, 'black',tuple((x + y)/2 for x, y in zip(top_left, bottom_right)),26,6)
    elif temp_board[3*row + col] == 1:
        pygame.draw.line(screen,'black', top_left, bottom_right, 8)
        pygame.draw.line(screen,'black',top_right, bottom_left, 8)
    
def draw_big_state(screen,board,big_row,big_col):
    if board.won_games[big_row*3 + big_col] != 0:
        if board.won_games[big_row*3 + big_col] == -1:
            pygame.draw.circle(screen, 'green', (big_row*WIDTH//3+121.5,big_col*HEIGHT//3+121.5),100,15)
        elif board.won_games[big_row*3 + big_col] == 1:
            pygame.draw.line(screen,'green',(big_row*WIDTH//3+29,big_col*HEIGHT//3+18),(big_row*WIDTH//3+214,big_col*HEIGHT//3+225),21)
            pygame.draw.line(screen,'green',(big_row*WIDTH//3+214,big_col*HEIGHT//3+18),(big_row*WIDTH//3+29,big_col*HEIGHT//3+225),21)

def drawGameState(screen, board):
    screen.fill(pygame.Color('white'))
    draw_big_grid(screen)
    for big_row in range(3):
        for big_col in range(3):
            draw_little_grid_and_next_moves(screen,board,big_row,big_col)     
            for row in range(3):
                for col in range(3):
                    draw_little_state(screen,board,big_row,big_col,row,col)
            draw_big_state(screen,board,big_row,big_col)