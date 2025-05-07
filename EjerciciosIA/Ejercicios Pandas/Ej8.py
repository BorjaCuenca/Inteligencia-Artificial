import pandas as pd
df = pd.read_csv('bmw.csv')

print ("Número de filas: ", df.shape[0])
print ("Número de columnas: ", df.shape[1])
print ("Antepenúltimo registro: \n", df.iloc[-2])
