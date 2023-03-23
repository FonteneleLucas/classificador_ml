from pydantic import BaseModel

class ClassificadorResponse(BaseModel):
    renda_mensal: float
    valor_divida: float
    num_parc: int
    valor_parc: float
    valor_aprox: float
