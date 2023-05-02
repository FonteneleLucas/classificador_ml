import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# carrega o conjunto de dados
data = pd.read_csv('arquivo.csv')

# separa o target das features
X = data.drop('NUM_PARC', axis=1)
y = data['NUM_PARC']

# separa o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instancia o algoritmo KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# treina o modelo com o conjunto de treino
model.fit(X_train, y_train)

# faz a previsão com o conjunto de teste
y_pred = model.predict(X_test)

# verifica a acurácia do modelo
acc = accuracy_score(y_test, y_pred)
print('Acurácia:', acc)

# cria um DataFrame com os valores reais, as previsões e a acurácia
result = pd.DataFrame({'real': y_test, 'predito': y_pred, 'acuracia': [acc] * len(y_test)})

# salva o DataFrame em um arquivo CSV
result.to_csv('resultado.csv', index=False)
