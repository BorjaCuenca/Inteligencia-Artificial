import pandas as pd
df = pd.read_csv('bmw.csv')

sub_df = df[['mileage', 'price', 'mpg']]
print (sub_df.sample(frac=0.2))