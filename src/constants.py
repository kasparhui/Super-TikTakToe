WIDTH = 729
HEIGHT = 729
# Minimax AI: it looks AI_DEPTH - 1 moves ahead, but stops deepening early when
# a move would take much longer than AI_TIME_LIMIT seconds.
# Set AI_TIME_LIMIT = None to always search the full depth.
AI_DEPTH = 14
AI_TIME_LIMIT = 2.0
tic_tac_toe_mapping = {'X': 1,'O': -1,'-': 0}
inverse_tic_tac_toe_mapping = {value: key for key, value in tic_tac_toe_mapping.items()}
short_mapping = {'X': 'O','O': 'X'}
def arrange_array(arr):
    arr = arr.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    return arr