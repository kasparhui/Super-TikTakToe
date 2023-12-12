import numpy as np
#from itertools import product
#import torch

a = np.array([1,2,3])
b = np.array([1,2,3])
#print(a * 2)

ggt = lambda a, b: ggt(b, a % b) if b else a

print(ggt(12, 8))