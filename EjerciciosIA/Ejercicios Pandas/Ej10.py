import pandas as pd
df = pd.read_csv('bmw.csv')

print (df[(df['mileage'] < 10000) & (df['mpg'] > 40)])