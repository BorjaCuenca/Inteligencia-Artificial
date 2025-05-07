import numpy as np

def invertRowsOrder(array):
    return np.flip(array, axis=0)

array = np.arange(9).reshape(3,3)
print(invertRowsOrder(array))