import pandas as pd
dataset = 'Breast Cancer Wisconsin.data'
df = pd.read_csv(dataset)
df = df.iloc[:, 1:]
df.to_csv('sem1coluna.csv', index=False)

#Nos datasets Breast Cancer Wisconsin e Darwin a primeira coluna foi removida pois representavam o ID.