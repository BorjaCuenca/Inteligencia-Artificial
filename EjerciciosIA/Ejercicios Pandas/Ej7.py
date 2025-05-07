import pandas as pd
df = pd.read_csv('bmw.csv')

print("Media: ", df['engineSize'].mean())
print("Desviación típica: ", df['engineSize'].std())
print("Valor máximo: ", df['engineSize'].max())
print("Valor mínimo: ", df['engineSize'].min())

