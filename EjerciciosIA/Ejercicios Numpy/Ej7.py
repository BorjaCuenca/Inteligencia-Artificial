import numpy as np

array = np.arange(9)
#array = np.arange(10)

num_elements = np.size(array)
dim = int(np.sqrt(num_elements))

if dim*dim == num_elements:
    print(np.reshape(array, (dim,dim)))
else:
    print("No se puede trasnformar en una matríz cuadrada.")