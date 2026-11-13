"""Prueba rápida de cualquier modelo ONNX: sin restricción y con restricción.

Usa la plantilla de chat que trae el propio modelo, así que sirve para Phi, Qwen, etc.

Uso:  python scripts/05_prueba_modelo.py RUTA_DEL_MODELO
      (sin argumento usa la variable MODEL_DIR)
"""
import json
import os
import sys
import time

import onnxruntime_genai as og
from jsonschema import validate

ruta = sys.argv[1] if len(sys.argv) > 1 else os.environ["MODEL_DIR"]
model = og.Model(ruta)
tokenizer = og.Tokenizer(model)

schema = {
    "type": "object",
    "properties": {
        "entidad": {"type": "string", "minLength": 2, "maxLength": 80},
        "riesgo": {"type": "string", "enum": ["ALTO", "MEDIO", "BAJO"]},
        "puntuacion": {"type": "integer", "minimum": 1, "maximum": 100},
        "fecha_revision": {"type": "string", "format": "date"},
    },
    "required": ["entidad", "riesgo", "puntuacion", "fecha_revision"],
    "additionalProperties": False,
}

documento = (
    "Informe interno: Contoso S.A. presenta deficiencias graves en el control "
    "de accesos. Revisión realizada el 3 de marzo de 2026."
)
mensajes = [{
    "role": "user",
    "content": "Extrae del siguiente documento la entidad, el nivel de riesgo "
               "(ALTO, MEDIO o BAJO), una puntuación de 1 a 100 y la fecha de revisión. "
               "Responde en JSON con los campos entidad, riesgo, puntuacion y "
               "fecha_revision.\n\n" + documento,
}]
prompt = tokenizer.apply_chat_template(json.dumps(mensajes), add_generation_prompt=True)
entrada = tokenizer.encode(prompt)


def generar(restringir: bool) -> None:
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(entrada) + 200)
    if restringir:
        params.set_guidance("json_schema", json.dumps(schema))
    generador = og.Generator(model, params)
    generador.append_tokens(entrada)
    stream = tokenizer.create_stream()
    salida, n, inicio = "", 0, time.time()
    while not generador.is_done():
        generador.generate_next_token()
        salida += stream.decode(generador.get_next_tokens()[0])
        n += 1
    segundos = time.time() - inicio
    print(f"\n== {'Con' if restringir else 'Sin'} restricción "
          f"({n} tokens, {n / segundos:.1f} tokens/s)")
    print(salida)
    try:
        validate(json.loads(salida), schema)
        print("-> Válido contra el esquema")
    except Exception as e:
        print(f"-> NO válido: {type(e).__name__}: {str(e)[:150]}")


print(f"Modelo: {ruta}")
generar(restringir=False)
generar(restringir=True)