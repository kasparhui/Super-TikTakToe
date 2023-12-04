import os

WIDTH = 729
HEIGHT = 729
tic_tac_toe_mapping = {'X': 1,'O': -1,'-': 0}
inverse_tic_tac_toe_mapping = {value: key for key, value in tic_tac_toe_mapping.items()}
input_size = 9
hidden_size1 = 300
hidden_size2 = 100
hidden_size3 = 50
hidden_size4 = 30
hidden_size5 = 20
hidden_size6 = 5
output_size = 1


input_short_size = 9
hidden_short_size1 = 120
hidden_short_size2 = 90
hidden_short_size3 = 35
output_short_size = 1
path = os.path.join('models', f'trained_model_weights_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{output_size}.pth')
path_short = os.path.join('models', f'trained_model_weights_short_{input_short_size}_{hidden_short_size1}_{hidden_short_size2}_{hidden_short_size3}_{output_short_size}.pth')