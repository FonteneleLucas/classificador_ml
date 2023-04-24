import pandas as pd
from sklearn.preprocessing import StandardScaler

# Lê o arquivo CSV e armazena em um DataFrame
df = pd.read_csv('dataset_gerado.csv')

# Remove as linhas com valores faltantes
# df = df.dropna()

# Converte as variáveis categóricas em variáveis dummy
df = pd.get_dummies(df, columns=['SEGMENTO', 'SITUACAO', 'CANAL'])

# Aplica a normalização nas variáveis numéricas
scaler = StandardScaler()
df[['VALOR_RENDA', 'SALDO_DEVEDOR']] = scaler.fit_transform(df[['VALOR_RENDA', 'SALDO_DEVEDOR']])

# Divide o dataset em variáveis explicativas (X) e alvo (y)
X = df.drop('NUM_PARCELAS', axis=1)
y = df['NUM_PARCELAS']

# Salva os arquivos CSV pré-processados
X.to_csv('dataset_treino.csv', index=False)
y.to_csv('dataset_teste.csv', index=False)


print("#### - DATASET's GERADOS - ####")
