import pandas as pd
df = pd.read_csv('bmw.csv')

print(df['model'].replace(r'(\d+) Series', r'Series \1', regex=True))