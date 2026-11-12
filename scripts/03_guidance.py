import os
from guidance import models, json as gen_json
from pydantic import BaseModel, Field

class Evaluacion(BaseModel):
    entidad: str
    riesgo: str
    puntuacion: int = Field(ge=1, le=100)

lm = models.OnnxRuntimeGenAI(os.environ["MODEL_DIR"])
lm += "Evalúa el riesgo de Contoso S.A. en formato JSON:\n"
lm += gen_json(name="resultado", schema=Evaluacion)
print("\n--- RESULTADO ---")
print(lm["resultado"])
