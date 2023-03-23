from pydantic import BaseModel

class ClassificadorRequest(BaseModel):
    renda_mensal: float
    valor_divida: float
