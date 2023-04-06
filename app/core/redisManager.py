import pickle
import pandas as pd
import numpy as np
import redis
from sklearn.neighbors import NearestNeighbors

class RedisManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_conn = redis.Redis(host=host, port=port, db=db)

    def salvar_modelo(self, _name_model, _file, columns, _n_neighbors=3):
        data = pd.read_csv(_file, usecols=columns)

        array = data.values
        neigh = NearestNeighbors(n_neighbors=_n_neighbors)
        neigh.fit(array)
        model_bytes = pickle.dumps(neigh)

        self.redis_conn.set(_name_model, model_bytes)

    def carregar_modelo(self, _name_model):
        try:
            model_bytes = self.redis_conn.get(_name_model)
            model = pickle.loads(model_bytes)
            return model
        except:
            return None