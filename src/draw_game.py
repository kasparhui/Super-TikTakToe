import pygame
from constants import *


def drawGameState(screen, board):
    screen.fill(pygame.Color('white'))
    for i in range(1,3):
        pygame.draw.rect(screen, 'black', pygame.Rect(i*WIDTH//3-10, 9, 20, WIDTH-18))
        pygame.draw.rect(screen, 'black', pygame.Rect(9, i*HEIGHT//3-10, HEIGHT-18, 20))
    
    for big_row in range(3):
        for big_col in range(3):
            for i in range(1,3):
                pygame.draw.rect(screen, 'black', pygame.Rect(i*(WIDTH//3-36)//3+big_row*WIDTH//3+13, big_col*HEIGHT//3+18, 10, WIDTH//3-36))
                pygame.draw.rect(screen, 'black', pygame.Rect(big_row*WIDTH//3+18, i*(HEIGHT//3-36)//3 + big_col*HEIGHT//3+13, HEIGHT//3-36, 10))
            for row in range(3):
                for col in range(3):
                    temp_board = board[big_row*3 + big_col]
                    if temp_board[3*row + col] == 'O':
                        pygame.draw.circle(screen, 'black', (big_row*WIDTH//3+row*((WIDTH//3-36)//3)+52.5,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+52.5),26)
                        pygame.draw.circle(screen, 'white', (big_row*WIDTH//3+row*((WIDTH//3-36)//3)+52.5,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+52.5),20)
                    elif temp_board[3*row + col] == 'X':
                        pygame.draw.line(screen,'black',(big_row*WIDTH//3+row*((WIDTH//3-36)//3)+32,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+26),(big_row*WIDTH//3+row*((WIDTH//3-36)//3)+73,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+79),8)
                        pygame.draw.line(screen,'black',(big_row*WIDTH//3+row*((WIDTH//3-36)//3)+32,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+79),(big_row*WIDTH//3+row*((WIDTH//3-36)//3)+73,big_col*HEIGHT//3+col*((HEIGHT//3-36)//3)+26),8)