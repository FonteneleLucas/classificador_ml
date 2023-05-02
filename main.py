import pandas as pd
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('arquivo.csv')

data.drop(['coluna1', 'coluna2'], axis=1, inplace=True)

X = data.drop('NUM_PARC', axis=1)
y = data['NUM_PARC']

model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')

model.fit(X)

distances, indices = model.kneighbors(X)

predictions = pd.DataFrame({'predicao': y[indices[:, 0]]})

actuals = pd.DataFrame({'real': y})

result = pd.concat([predictions, actuals], axis=1)
result.to_csv('comparacao.csv', index=False)
