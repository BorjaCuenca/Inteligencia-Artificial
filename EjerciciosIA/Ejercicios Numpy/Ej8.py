import numpy as np

array = np.random.randint(0, 10, size=(3,3))
print (array, "\n")
print(np.max(array, axis=0))