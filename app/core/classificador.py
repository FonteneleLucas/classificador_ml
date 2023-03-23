from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class Classificador:
    def classificar(self, renda_mensal: float, valor_divida: float) -> float:
        
        data = pd.read_csv('negociacoes.csv')

        array = data[['RM', 'VD']].values

        pca = PCA(2)

        df = pca.fit_transform(array)

        # Definindo novo nó a ser classificado
        novo_no = np.array([[renda_mensal, valor_divida]])

        # Transformando o novo nó usando PCA
        novo_no_transformed = pca.transform(novo_no)

        kmeans = KMeans(n_clusters=6)

        # Prevendo as labels dos clusters
        label = kmeans.fit_predict(df)

        # Obtendo as labels únicas
        u_labels = np.unique(label)

        # Calculando a distância do novo nó para cada centroide
        centroides = kmeans.cluster_centers_
        distancias = cdist(novo_no_transformed, centroides)

        # Obtendo o índice do cluster mais próximo
        cluster_mais_proximo = np.argmin(distancias)

        # Adicionando as labels do cluster aos dados originais
        data['cluster'] = label

        # Salvando os dados em um arquivo CSV
        data.to_csv('negociacoes_clusterizado.csv', index=False)

        # Calculando os valores médios para as colunas Q e V por label de cluster
        cluster_means = data.groupby('cluster')['RM','VD','NP','VP'].mean()
        # print(cluster_means)

        # Obtendo os valores médios para o cluster mais próximo do novo nó
        rm_mean = cluster_means.loc[cluster_mais_proximo, 'RM']
        np_mean = cluster_means.loc[cluster_mais_proximo, 'NP']
        vd_mean = cluster_means.loc[cluster_mais_proximo, 'VD']
        vp_mean = cluster_means.loc[cluster_mais_proximo, 'VP']

        # Retornando os valores médios para NP, VP, VD para o novo nó
        print(f"O novo nó deve escolher RM = {rm_mean:.2f}, NP = {np_mean:.2f}, VP = {vp_mean:.2f}, VD = {vd_mean:.2f}")
        return rm_mean, vd_mean, np_mean, vp_mean