import pickle
import pandas as pd
import numpy as np
import redis
from sklearn.neighbors import NearestNeighbors


class Classificador:
    def __init__(self, redis_manager):
        self.redis_manager = redis_manager

    def classificar(self, renda_mensal: float, valor_divida: float) -> float:
        model = self.redis_manager.carregar_modelo('model')

        if model == None:
            return None

        # Definindo novo nó a ser classificado
        novo_no = np.array([[renda_mensal, valor_divida]])

        # Obtendo os índices dos vizinhos mais próximos ao novo nó
        nn_indices = model.kneighbors(novo_no, return_distance=False)

        # Obtendo os valores de NP, VP e VD para os vizinhos mais próximos
        data = pd.read_csv('negociacoes.csv')
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

    def salvarModelo(self):
        try:
            self.redis_manager.salvar_modelo('model', 'negociacoes.csv', ['RM', 'VD'])
            return "Modelo salvo no REDIS"
        except:
            return "Erro ao salvar no REDIS"
