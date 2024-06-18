import pandas as pd
df = pd.read_csv('darwin.csv')
df.to_csv('file_no_header.csv', index=False, header=False)

#Nos datasets Darwin e Matetrnal Health Risk o cabeçalho foi removido.