import pygame

import game
from AI import create_ai, ai_move
from constants import *
from draw_game import drawGameState

# ------------------------------- Settings -------------------------------
# Who plays X and who plays O. 'type' is one of:
#   'human'       you play with the mouse
#   'minimax'     minimax with alpha-beta pruning. Options:
#                   'depth'       looks depth - 1 moves ahead
#                   'time_limit'  rough thinking time per move in seconds; the AI
#                                 stops deepening early when it runs out.
#                                 Leave it out to always search the full depth.
#   'mcts'        Monte Carlo tree search with random playouts. Option:
#                   'iterations'  random games per move (default 200000)
#   'neural_net'  AlphaZero: tree search guided by the network trained with
#                 train.py. Options:
#                   'model'        'small', 'medium' or 'large' (default 'medium')
#                   'simulations'  positions the network looks at per move
#                                  (default 400; more = stronger and slower)
#   'random'      random moves
# Example: 'O': {'type': 'neural_net', 'model': 'medium', 'simulations': 800}
PLAYERS = {
    'O': {'type': 'human'},
    'X': {'type': 'neural_net', 'model': 'medium', 'simulations': 4000},
}
STARTING_PLAYER = 'O'
# ------------------------------------------------------------------------

TEXT_HEIGHT = 70


def new_board():
    return game.Board(playerturn=tic_tac_toe_mapping[STARTING_PLAYER])


def play(board, position):
    board.execute(position)
    board.last_move = position


def little_position(pos):
    if 18 <= pos < 87:
        return 0
    elif 87 <= pos <= 156:
        return 1
    elif 156 < pos <= 225:
        return 2
    else:
        return None


def position_from_mouse(mouse_position):
    """(big field, little field) under the mouse, or None if it is on a grid line or outside the board."""
    x, y = mouse_position
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return None
    cell_x = little_position(x % (WIDTH // 3))
    cell_y = little_position(y % (HEIGHT // 3))
    if cell_x is None or cell_y is None:
        return None
    big_x = x // (WIDTH // 3)
    big_y = y // (HEIGHT // 3)
    return (3 * big_y + big_x, 3 * cell_y + cell_x)


def undo(board, ais):
    """Take back the last move; against an AI also the AI's answer, so that it is the human's turn again."""
    human_players = sum(ai is None for ai in ais.values())
    if human_players == 2:
        moves_to_undo = 1
    elif human_players == 1:
        human_to_move = ais[inverse_tic_tac_toe_mapping[board.playerturn]] is None
        moves_to_undo = 2 if human_to_move else 1
    else:
        return
    if len(board.log) >= moves_to_undo:
        for _ in range(moves_to_undo):
            board.undo_move()


def status_text(board):
    if not board.game_is_over:
        return f'Spieler {inverse_tic_tac_toe_mapping[board.playerturn]} ist am Zug'
    if game.game_is_win(board.won_games)[0]:
        # the player who made the last move has won
        return f'Spieler {inverse_tic_tac_toe_mapping[-board.playerturn]} hat gewonnen!'
    return 'Unentschieden!'


def draw_win_line(screen, board):
    win_line = game.game_is_win(board.won_games)[1]
    if 1 <= win_line <= 3:      # column
        x = (win_line - 1) * WIDTH // 3 + WIDTH // 6
        pygame.draw.line(screen, 'red', (x, 0), (x, HEIGHT), 15)
    elif 4 <= win_line <= 6:    # row
        y = (win_line - 4) * HEIGHT // 3 + HEIGHT // 6
        pygame.draw.line(screen, 'red', (0, y), (WIDTH, y), 15)
    elif win_line == 7:
        pygame.draw.line(screen, 'red', (0, 0), (WIDTH, HEIGHT), 20)
    elif win_line == 8:
        pygame.draw.line(screen, 'red', (0, HEIGHT), (WIDTH, 0), 20)


def draw(screen, font, board):
    drawGameState(screen, board)
    if board.game_is_over:
        draw_win_line(screen, board)
    text_surface = font.render(status_text(board), True, 'black')
    text_rect = text_surface.get_rect()
    text_rect.center = (WIDTH // 2, HEIGHT + 25)
    screen.blit(text_surface, text_rect)
    pygame.display.flip()


def play_game():
    """Mouse click: play a move.  U: undo.  Space (after the game): new game."""
    pygame.init()
    pygame.display.set_caption('Super TikTakToe')
    screen = pygame.display.set_mode((WIDTH, HEIGHT + TEXT_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 80)

    ais = {symbol: create_ai(symbol, PLAYERS[symbol]) for symbol in ('X', 'O')}
    board = new_board()

    running = True
    while running:
        for e in pygame.event.get():
            symbol = inverse_tic_tac_toe_mapping[board.playerturn]
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if ais[symbol] is None and not board.game_is_over:
                    position = position_from_mouse(e.pos)
                    if position is not None and board.is_valid_move(position):
                        play(board, position)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_u:
                    undo(board, ais)
                elif e.key == pygame.K_SPACE and board.game_is_over:
                    board = new_board()
        if not running:
            break

        # Draw first, so the last move is visible while the AI is thinking.
        draw(screen, font, board)
        clock.tick(15)

        symbol = inverse_tic_tac_toe_mapping[board.playerturn]
        if ais[symbol] is not None and not board.game_is_over:
            position = ai_move(ais[symbol], PLAYERS[symbol], board)
            play(board, position)
            pygame.event.clear(pygame.MOUSEBUTTONDOWN)  # ignore clicks made while the AI was thinking
            print(f"{symbol} ({PLAYERS[symbol]['type']}) spielt {tuple(int(i) for i in position)}")

    pygame.quit()


if __name__ == '__main__':
    play_game()
