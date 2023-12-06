def check_rows(little_board):
    a = (2 != little_board[0]== little_board[1]==little_board[2] !=0)
    b = (2 != little_board[3]==little_board[4]==little_board[5] !=0) 
    c = (2 != little_board[6]==little_board[7]==little_board[8] !=0)
    if a: return True, 4
    elif b: return True, 5
    elif c: return True, 6
    else: return False, -1

def check_column(little_board):
    a = (2 != little_board[0]==little_board[3]==little_board[6]!=0)
    b = (2 != little_board[1]==little_board[4]==little_board[7]!=0) 
    c = (2 != little_board[2]==little_board[5]==little_board[8]!=0)
    if a: return True, 1
    elif b: return True, 2
    elif c: return True, 3

    else: return False, -1
def check_diagonal(little_board):
    a = (2 != little_board[0]==little_board[4]==little_board[8]!=0) 
    b = (2 != little_board[2]==little_board[4]==little_board[6]!=0)
    if a: return True, 7
    elif b: return True, 8
    else: return False, -1

def game_is_tie(little_board):
    for i in little_board:
        if i==0: return False, -1
    else: return True, 0

def game_is_win(little_board):
    row_winner=check_rows(little_board)
    column_winner=check_column(little_board)
    diagonal_winner=check_diagonal(little_board)
    if row_winner[0]:
        return True, row_winner[1]
    elif column_winner[0]:
        return True, column_winner[1]
    elif diagonal_winner[0]:
        return True, diagonal_winner[1]
    else: return False, 0

def check_if_game_over(little_board):
    if game_is_tie(little_board)[0]: return (True)
    is_win=game_is_win(little_board)[0]
    if is_win: 
        return (True)
    else: return False