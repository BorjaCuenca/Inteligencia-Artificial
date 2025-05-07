import pandas as pd
df = pd.read_csv('bmw.csv')

print(df.sample(frac=0.4))