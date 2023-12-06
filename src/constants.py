import os

WIDTH = 729
HEIGHT = 729
tic_tac_toe_mapping = {'X': 1,'O': -1,'-': 0}
inverse_tic_tac_toe_mapping = {value: key for key, value in tic_tac_toe_mapping.items()}
def arrange_array(arr):
    arr = arr.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    return arr