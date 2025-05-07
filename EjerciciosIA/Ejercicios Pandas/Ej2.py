import pandas as pd
df = pd.read_csv('bmw.csv')

print(df['year'], "\n")
print("Type: ", df['year'].dtype, " ", "Lenght: ", df['year'].count())