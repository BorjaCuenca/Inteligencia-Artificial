import pandas as pd
df = pd.read_csv('bmw.csv')

print(df.sort_values(by='mpg'))