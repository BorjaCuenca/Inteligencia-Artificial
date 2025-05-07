import numpy as np

def invertOrder(array):
    return np.flip(array)

array = range(0,20,2)
print(invertOrder(array))