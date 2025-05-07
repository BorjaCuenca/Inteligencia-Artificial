import numpy as np

array = np.random.randint(0, 10, size=(4,3))
print (array)
print(np.mean(array, axis=0))