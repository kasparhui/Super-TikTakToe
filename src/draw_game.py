import pygame
from constants import *

# A big field is addressed by (big_x, big_y) = (column, row) on the screen and
# has the index big_x + 3 * big_y in board.board / board.won_games. The cells
# inside a little board work the same way: index cell_x + 3 * cell_y.

BIG = WIDTH // 3            # size of one big field in pixels
CELL = (BIG - 36) // 3      # size of one little cell in pixels
HIGHLIGHT = (243, 219, 214)
LAST_MOVE = (61, 89, 171)


def draw_big_grid(screen):
    for i in range(1,3):
        pygame.draw.rect(screen, 'black', pygame.Rect(i*WIDTH//3-10, 9, 20, WIDTH-18))
        pygame.draw.rect(screen, 'black', pygame.Rect(9, i*HEIGHT//3-10, HEIGHT-18, 20))

def draw_little_grid_and_next_moves(screen, board, big_x, big_y):
    index = big_x + 3*big_y
    left, top = big_x*BIG, big_y*BIG
    # highlight the little boards the next move may be played in
    if not board.game_is_over and board.won_games[index] == 0 and board.next_move in (None, index):
        pygame.draw.rect(screen, HIGHLIGHT, pygame.Rect(left+10, top+10, BIG-20, BIG-20))
    for i in range(1,3):
        pygame.draw.rect(screen, 'black', pygame.Rect(left + i*CELL + 13, top+18, 10, BIG-36))
        pygame.draw.rect(screen, 'black', pygame.Rect(left+18, top + i*CELL + 13, BIG-36, 10))

def draw_little_state(screen, board, big_x, big_y, cell_x, cell_y):
    index = big_x + 3*big_y
    cell = cell_x + 3*cell_y
    value = board.board[index][cell]
    if value == 0:
        return
    left = big_x*BIG + cell_x*CELL + 32
    top = big_y*BIG + cell_y*CELL + 26
    right, bottom = left + 41, top + 53
    is_last_move = board.last_move is not None and board.last_move[0] == index and board.last_move[1] == cell
    color = LAST_MOVE if is_last_move else 'black'
    if value == -1:
        pygame.draw.circle(screen, color, ((left+right)/2, (top+bottom)/2), 26, 6)
    else:
        pygame.draw.line(screen, color, (left, top), (right, bottom), 8)
        pygame.draw.line(screen, color, (right, top), (left, bottom), 8)

def draw_big_state(screen, board, big_x, big_y):
    winner = board.won_games[big_x + 3*big_y]
    left, top = big_x*BIG, big_y*BIG
    if winner == -1:
        pygame.draw.circle(screen, 'green', (left+121.5, top+121.5), 100, 15)
    elif winner == 1:
        pygame.draw.line(screen, 'green', (left+29, top+18), (left+214, top+225), 21)
        pygame.draw.line(screen, 'green', (left+214, top+18), (left+29, top+225), 21)

def drawGameState(screen, board):
    screen.fill(pygame.Color('white'))
    draw_big_grid(screen)
    for big_y in range(3):
        for big_x in range(3):
            draw_little_grid_and_next_moves(screen, board, big_x, big_y)
            for cell_y in range(3):
                for cell_x in range(3):
                    draw_little_state(screen, board, big_x, big_y, cell_x, cell_y)
    # The big crosses and circles come last, so that no little grid is drawn over them.
    for big_y in range(3):
        for big_x in range(3):
            draw_big_state(screen, board, big_x, big_y)
