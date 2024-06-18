import pandas as pd

df = pd.read_csv('wdbc.data', header=None)
columns = df.columns.tolist()
columns.append(columns.pop(1))
df_reorder = df[columns]
df_reorder.to_csv('file_reordered.data', index=False, header=False)

#No Dataset Breast Cancer Wisconsin foi necessário mover a coluna de classes para o final, tal código tira ela da posição original (segunda coluna) e a move para a última.