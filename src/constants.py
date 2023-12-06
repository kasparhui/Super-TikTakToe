import os

WIDTH = 729
HEIGHT = 729
tic_tac_toe_mapping = {'X': 1,'O': -1,'-': 0}
inverse_tic_tac_toe_mapping = {value: key for key, value in tic_tac_toe_mapping.items()}
def arrange_array(arr):
    arr = arr.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    return arr
input_size = 9
hidden_size1 = 300
hidden_size2 = 100
hidden_size3 = 75
hidden_size4 = 30
hidden_size5 = 20
hidden_size6 = 5
output_size = 1
combined_size =[input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, output_size]
combined_size = [9, 20, 8, 1]
string_path = ''
for size in combined_size:
    string_path += f'_{size}'

input_short_size = 9
hidden_short_size1 = 120
hidden_short_size2 = 90
hidden_short_size3 = 35
output_short_size = 1
path = os.path.join('models', f'trained_model_weights{string_path}.pth')

path_short = os.path.join('models', f'trained_model_weights_short_{input_short_size}_{hidden_short_size1}_{hidden_short_size2}_{hidden_short_size3}_{output_short_size}.pth')