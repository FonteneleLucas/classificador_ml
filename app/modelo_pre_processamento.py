import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# carrega o dataset
df = pd.read_csv('seu_dataset.csv')

# lidar com dados faltantes
df.fillna(df.mean(), inplace=True)

# lidar com dados categóricos
df = pd.get_dummies(df, columns=['SEGMENTO', 'CANAL'])
label_encoder = LabelEncoder()
df['SITUACAO'] = label_encoder.fit_transform(df['SITUACAO'])

# normalizar os dados numéricos
scaler = MinMaxScaler()
df[['VALOR_RENDA', 'SALDO_DEVEDOR']] = scaler.fit_transform(df[['VALOR_RENDA', 'SALDO_DEVEDOR']])

# separar os dados de treino e teste
X = df.drop('SITUACAO', axis=1)
y = df['SITUACAO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
