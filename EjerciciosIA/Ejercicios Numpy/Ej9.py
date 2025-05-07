import numpy as np

array = np.random.randint(0, 10, size=(3,3))
print(array, "\n")
occurrences = np.unique(array, return_counts=True)
print (np.array(occurrences))
