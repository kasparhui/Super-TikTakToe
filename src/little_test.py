import numpy as np
from itertools import product

possible_values = [-1, 0, 1]
combinations = list(map(list, product(possible_values, repeat=9)))
print(combinations)
print(len(combinations))
