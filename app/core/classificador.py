import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class Classificador:
    def classificar(self, renda_mensal: float, valor_divida: float) -> float:

        # Lendo os dados e separando em X e y
        data = pd.read_csv('negociacoes.csv')
        X = data[['RM', 'VD']].values
        y = data[['NP', 'VP']].values

        # Dividindo o conjunto de dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizando os dados de entrada do conjunto de treino
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Normalizando os dados de entrada do conjunto de teste utilizando a mesma normalização do conjunto de treino
        X_test = scaler.transform(X_test)

        # Criando o modelo de rede neural
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(2)
        ])

        # Compilando o modelo com a função de perda MSE e o otimizador Adam
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Treinando o modelo com o conjunto de treino
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # Avaliando o modelo com o conjunto de teste
        loss = model.evaluate(X_test, y_test)
        print(f"Perda no conjunto de teste: {loss:.2f}")

        # Utilizando o modelo para fazer previsões para o novo nó
        novo_no = np.array([[renda_mensal, valor_divida]])
        novo_no = scaler.transform(novo_no)
        np_mean, vp_mean = model.predict(novo_no)[0]

        # Retornando os valores médios para NP, VP para o novo nó
        print(f"Plano sugerido: NP = {np_mean:.2f}, VP = {vp_mean:.2f}")
        return renda_mensal, valor_divida, np_mean, vp_mean
