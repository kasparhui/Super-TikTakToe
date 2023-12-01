import numpy as np

# Zwei Beispielarrays
array1 = np.array([100, 80, 60 ,50 ,25, 15, 45, 70, 75, 10, 65, 95, 55, 5, 20, 90, 85, 30, 35, 40])
array2 = np.array([1  , 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,15, 16, 17, 18, 19, 20])

# Indizes der größten Elemente im ersten Array finden
max_indices = np.argsort(array1)[-10:]

# Die entsprechenden Elemente aus dem zweiten Array auswählen
result = array2[max_indices]

print(result)
