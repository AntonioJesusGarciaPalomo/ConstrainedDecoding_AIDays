import os, sys, json
import onnxruntime_genai as og
from jsonschema import validate

RESTRINGIR = "--libre" not in sys.argv

model = og.Model(os.environ["MODEL_DIR"])
tokenizer = og.Tokenizer(model)
stream = tokenizer.create_stream()

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
prompt = (
    "<|user|>\nExtrae del siguiente documento la entidad, el nivel de riesgo "
    "(ALTO, MEDIO o BAJO), una puntuación de 1 a 100 y la fecha de revisión. "
    "Responde en JSON con los campos entidad, riesgo, puntuacion y "
    f"fecha_revision.\n\n{documento}<|end|>\n<|assistant|>\n"
)
entrada = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=len(entrada) + 200)
if RESTRINGIR:
    params.set_guidance("json_schema", json.dumps(schema))

generator = og.Generator(model, params)
generator.append_tokens(entrada)

salida = ""
while not generator.is_done():
    generator.generate_next_token()
    salida += stream.decode(generator.get_next_tokens()[0])

print("Con restricción:" if RESTRINGIR else "Sin restricción:")
print(salida)
try:
    validate(json.loads(salida), schema)
    print("-> Válido contra el esquema")
except Exception as e:
    print(f"-> NO válido: {type(e).__name__}: {str(e)[:200]}")
