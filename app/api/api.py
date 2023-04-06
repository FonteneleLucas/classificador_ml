from flask import request, jsonify
from core.classificador import Classificador
from .entrada.classificadorRequest import ClassificadorRequest
from .saida.classificadorResponse import ClassificadorResponse
import pandas as pd
import numpy as np
import redis
from core.redisManager import RedisManager


class API:
    def __init__(self, app):
        self.redis_manager = RedisManager()
        self.app = app
        self.classificador = Classificador(self.redis_manager)
        self._add_routes()

    def _add_routes(self):
        self.app.add_url_rule('/api/classificar', view_func=self._calcular, methods=['POST'])
        self.app.add_url_rule('/api/salvarModelo', view_func=self._salvar_modelo, methods=['POST'])

    def _calcular(self):
        data = ClassificadorRequest.parse_raw(request.data)
        resultado = self.classificador.classificar(data.renda_mensal, data.valor_divida)
        if resultado == None:
            return "Modelo não encontrado no Redis."
        else:
            output = ClassificadorResponse(
                renda_mensal = resultado[0], 
                valor_divida = resultado[1], 
                num_parc = resultado[2], 
                valor_parc = resultado[3],
                valor_aprox = resultado[2] * resultado[3]
            )
            return output.dict()

    def _salvar_modelo(self):
        resultado = self.classificador.salvarModelo()
        return resultado