from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class Classificador:
    def classificar(self, renda_mensal: float, valor_divida: float) -> float:
        
        data = pd.read_csv('negociacoes.csv')

        array = data[['RM', 'VD']].values

        # Normalizar os dados
        scaler = StandardScaler()
        array_norm = scaler.fit_transform(array)

        # Normalizar o novo nó
        novo_no = scaler.transform([[renda_mensal, valor_divida]])

        # Definindo o objeto NearestNeighbors com a métrica 'cosine'
        neigh = NearestNeighbors(n_neighbors=3, metric='cosine')
        neigh.fit(array_norm)

        # Obtendo os índices dos vizinhos mais próximos ao novo nó
        nn_indices = neigh.kneighbors(novo_no, return_distance=False)

        # Obtendo os valores de NP, VP e VD para os vizinhos mais próximos
        rm_values = data.loc[nn_indices[0], 'RM']
        vd_values = data.loc[nn_indices[0], 'VD']
        np_values = data.loc[nn_indices[0], 'NP']
        vp_values = data.loc[nn_indices[0], 'VP']

        # Obtendo os valores médios de NP, VP e VD para os vizinhos mais próximos
        rm_mean = np.mean(rm_values)
        vd_mean = np.mean(vd_values)
        np_mean = np.mean(np_values)
        vp_mean = np.mean(vp_values)

        # Retornando os valores médios para NP, VP, VD para o novo nó
        print(f"Plano sugerido: RM = {rm_mean:.2f}, VD = {vd_mean:.2f}, NP = {np_mean:.2f}, VP = {vp_mean:.2f}")
        return rm_mean, vd_mean, np_mean, vp_mean
