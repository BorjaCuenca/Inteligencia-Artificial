import pandas as pd
df = pd.read_csv('bmw.csv')

print(df['mileage'][::7])