"""Contrato v2: admite «no consta» y tiene respuesta correcta conocida.

Lo comparten 06_calidad.py (evaluación) y 07_lora.py (entrenamiento), de modo
que el modelo se entrena y se evalúa con el mismo prompt y el mismo esquema.
"""
import json
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluacionV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entidad: str = Field(min_length=2, max_length=80)
    riesgo: Optional[Literal["ALTO", "MEDIO", "BAJO"]]   # null: el documento no lo evalúa
    fecha_revision: Optional[date]                        # null: no consta
    acciones: list[str] = Field(max_length=3)             # []: el documento no indica ninguna


SCHEMA = EvaluacionV2.model_json_schema()
SCHEMA_JSON = json.dumps(SCHEMA, ensure_ascii=False)

INSTRUCCIONES = (
    "Extrae la evaluación de riesgo del documento. Responde únicamente con un objeto "
    "JSON que cumpla este JSON Schema:\n" + SCHEMA_JSON + "\n\n"
    "Reglas:\n"
    "- entidad: el nombre de la entidad evaluada, tal como aparece en el documento.\n"
    "- riesgo: ALTO, MEDIO o BAJO según la gravedad de los hallazgos; null si el "
    "documento no evalúa el riesgo.\n"
    "- fecha_revision: la fecha de la revisión en formato AAAA-MM-DD; null si no consta.\n"
    "- acciones: las acciones que el documento exige o recomienda, copiadas del texto; "
    "lista vacía si no indica ninguna.\n\n"
    "Documento:\n"
)


def mensajes(texto: str) -> list[dict]:
    return [{"role": "user", "content": INSTRUCCIONES + texto}]


def respuesta(esperado: dict) -> str:
    """La salida correcta, en el orden de campos del contrato."""
    return json.dumps({k: esperado[k] for k in ("entidad", "riesgo", "fecha_revision", "acciones")},
                      ensure_ascii=False)