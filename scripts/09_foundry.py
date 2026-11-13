"""Prepara un modelo ONNX para cargarlo en Foundry Local como modelo propio.

Hace tres cosas:
  1. Escribe inference_model.json (nombre y plantilla de chat) dentro del modelo.
  2. Copia la carpeta del modelo al destino indicado, con el nombre elegido.
  3. Deja dos peticiones de prueba para la API: una normal y otra con JSON Schema.

Uso (dentro de WSL, desde la raíz del proyecto):
  python scripts/09_foundry.py models/qwen-0.5b-lora-int4 \
      --nombre qwen-evaluacion-lora \
      --destino /mnt/c/Users/TU_USUARIO/foundry-modelos
"""
import argparse
import json
import os
import shutil

from transformers import AutoTokenizer

from contrato_v2 import SCHEMA, mensajes

ap = argparse.ArgumentParser()
ap.add_argument("modelo", help="carpeta del modelo ONNX (la que contiene genai_config.json)")
ap.add_argument("--nombre", required=True, help="nombre con el que aparecerá en Foundry Local")
ap.add_argument("--destino", required=True, help="carpeta donde copiar el modelo")
ap.add_argument("--tokenizer", default="models/qwen-0.5b-lora",
                help="carpeta en formato Hugging Face de la que leer la plantilla de chat")
a = ap.parse_args()

if not os.path.exists(os.path.join(a.modelo, "genai_config.json")):
    raise SystemExit(f"No encuentro genai_config.json en {a.modelo}")

# 1. Plantilla de chat: la misma que usó el modelo al entrenar y al evaluar,
#    con {Content} donde irá el mensaje del usuario.
tok = AutoTokenizer.from_pretrained(a.tokenizer)
plantilla = tok.apply_chat_template([{"role": "user", "content": "{Content}"}],
                                    tokenize=False, add_generation_prompt=True)
inference = {"Name": a.nombre, "PromptTemplate": {"assistant": "{Content}", "prompt": plantilla}}

# 2. Copia del modelo
carpeta = os.path.join(a.destino, a.nombre)
if os.path.exists(carpeta):
    shutil.rmtree(carpeta)
shutil.copytree(a.modelo, carpeta)
with open(os.path.join(carpeta, "inference_model.json"), "w", encoding="utf-8") as f:
    json.dump(inference, f, ensure_ascii=False, indent=2)

# 3. Peticiones de prueba: un documento sin fecha y sin acciones
documento = ("Se concluye que Boulder Innovations no cumple ninguno de los requisitos "
             "esenciales de conservación de datos; la exposición es máxima.")
base = {"model": a.nombre, "messages": mensajes(documento), "max_tokens": 200, "temperature": 0}
con_schema = dict(base, response_format={
    "type": "json_schema",
    "json_schema": {"name": "evaluacion", "strict": True, "schema": SCHEMA},
})
for fichero, cuerpo in (("peticion.json", base), ("peticion_schema.json", con_schema)):
    with open(os.path.join(a.destino, fichero), "w", encoding="utf-8") as f:
        json.dump(cuerpo, f, ensure_ascii=False, indent=2)

print(f"Modelo copiado en: {carpeta}")
print(f"Ficheros: {sorted(os.listdir(carpeta))}")
print("\nPlantilla de chat escrita en inference_model.json:")
print(plantilla)
print(f"\nPeticiones de prueba en {a.destino}: peticion.json y peticion_schema.json")
print("Respuesta correcta esperada: riesgo ALTO, fecha_revision null, acciones []")