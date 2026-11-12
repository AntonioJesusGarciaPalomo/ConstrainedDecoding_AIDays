"""Comparativa de estrategias de salida estructurada sobre el mismo contrato Pydantic.

A  prompt              1 intento, sin restricción
B  retry               hasta 3 intentos, reinyectando el error de Pydantic
C  constrained         1 intento, con set_guidance(json_schema)
D  constrained+retry   restricción + reintento para las reglas de negocio

Uso:  python scripts/04_comparativa.py          (dataset completo o ejemplos)
      python scripts/04_comparativa.py --n 3    (solo los 3 primeros documentos)
"""
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from typing import Literal

from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      ValidationInfo, field_validator, model_validator)

MAX_TOKENS_NUEVOS = 350
MAX_INTENTOS = 3


# ---------- Contrato: única fuente de verdad para las cuatro estrategias ----------
class Evaluacion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Parte estructural: se traduce a JSON Schema y la gramática puede imponerla.
    entidad: str = Field(min_length=2, max_length=80)
    riesgo: Literal["ALTO", "MEDIO", "BAJO"]
    puntuacion: int = Field(ge=1, le=100)
    fecha_revision: date
    acciones: list[str] = Field(max_length=3)

    # Parte de negocio: NO cabe en el JSON Schema. Solo la validación la detecta.
    @field_validator("entidad")
    @classmethod
    def entidad_aparece_en_documento(cls, v: str, info: ValidationInfo) -> str:
        doc = (info.context or {}).get("documento")
        if doc is not None and v.lower() not in doc.lower():
            raise ValueError("la entidad no aparece literalmente en el documento")
        return v

    @model_validator(mode="after")
    def coherencia_riesgo(self):
        if self.riesgo == "ALTO" and not self.acciones:
            raise ValueError("un riesgo ALTO exige al menos una acción")
        if self.riesgo == "ALTO" and self.puntuacion < 70:
            raise ValueError("un riesgo ALTO exige puntuacion >= 70")
        return self


SCHEMA = Evaluacion.model_json_schema()
SCHEMA_JSON = json.dumps(SCHEMA, ensure_ascii=False)

INSTRUCCIONES = (
    "Extrae la evaluación de riesgo del documento. Responde únicamente con un "
    "objeto JSON que cumpla este JSON Schema:\n" + SCHEMA_JSON + "\n\nDocumento:\n"
)

# Documentos de ejemplo. Si existe data/documentos.jsonl, se usa ese dataset.
DOCUMENTOS_EJEMPLO = [
    "Informe interno: Contoso S.A. presenta deficiencias graves en el control de "
    "accesos. Revisión realizada el 3 de marzo de 2026. Se exige rotar credenciales.",
    "Auditoría de Fabrikam Ibérica (12/01/2026): sin incidencias relevantes. "
    "Controles adecuados y documentación completa.",
    "Northwind Traders: retrasos puntuales en la notificación de incidentes. "
    "Se recomienda revisar el procedimiento. Revisado el 20 de febrero de 2026.",
]


# ---------- Modelo ----------
_modelo = _tokenizer = None


def formatear(mensajes: list[dict]) -> str:
    """Chat template de Phi-3.5 (comprobar contra la model card)."""
    texto = "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in mensajes)
    return texto + "<|assistant|>\n"


def generar(mensajes: list[dict], schema_json: str | None = None) -> tuple[str, int]:
    """Devuelve (texto, tokens generados). Con schema_json aplica constrained decoding."""
    global _modelo, _tokenizer
    import onnxruntime_genai as og

    if _modelo is None:
        _modelo = og.Model(os.environ["MODEL_DIR"])
        _tokenizer = og.Tokenizer(_modelo)

    entrada = _tokenizer.encode(formatear(mensajes))
    params = og.GeneratorParams(_modelo)
    params.set_search_options(max_length=len(entrada) + MAX_TOKENS_NUEVOS)
    if schema_json is not None:
        params.set_guidance("json_schema", schema_json)

    generador = og.Generator(_modelo, params)
    generador.append_tokens(entrada)
    stream = _tokenizer.create_stream()
    texto, n = "", 0
    while not generador.is_done():
        generador.generate_next_token()
        texto += stream.decode(generador.get_next_tokens()[0])
        n += 1
    return texto, n


# ---------- Estrategias ----------
def extraer_json(texto: str) -> str:
    """Modo tolerante: se queda con lo que hay entre la primera '{' y la última '}'."""
    i, j = texto.find("{"), texto.rfind("}")
    return texto[i:j + 1] if i != -1 and j > i else texto


def validar(texto: str, documento: str) -> None:
    Evaluacion.model_validate_json(texto, context={"documento": documento})


def mensaje_error(e: ValidationError) -> str:
    lineas = "\n".join(
        f"- {'.'.join(str(p) for p in err['loc']) or 'objeto'}: {err['msg']}"
        for err in e.errors()
    )
    return ("Tu respuesta no es válida. Errores:\n" + lineas +
            "\nDevuelve únicamente el objeto JSON corregido, sin texto antes ni "
            "después y sin bloque markdown. Usa exactamente los nombres de campo "
            "del esquema.")


def ejecutar(documento: str, restringir: bool, max_intentos: int) -> dict:
    mensajes = [{"role": "user", "content": INSTRUCCIONES + documento}]
    r = {"estricto_primer_intento": False, "valido_primer_intento": False,
         "valido": False, "intentos": 0, "tokens": 0, "segundos": 0.0,
         "errores_primer_intento": [], "salidas": []}
    inicio = time.time()

    for intento in range(1, max_intentos + 1):
        texto, n = generar(mensajes, SCHEMA_JSON if restringir else None)
        r["intentos"], r["tokens"] = intento, r["tokens"] + n
        r["salidas"].append(texto)

        if intento == 1:  # ¿la salida en bruto es consumible tal cual?
            try:
                validar(texto, documento)
                r["estricto_primer_intento"] = True
            except ValidationError:
                pass

        try:  # validación tolerante: se limpia el envoltorio antes de validar
            validar(extraer_json(texto), documento)
            r["valido"] = True
            r["valido_primer_intento"] = intento == 1
            break
        except ValidationError as e:
            if intento == 1:
                r["errores_primer_intento"] = [err["type"] for err in e.errors()]
            mensajes += [{"role": "assistant", "content": texto},
                         {"role": "user", "content": mensaje_error(e)}]

    r["segundos"] = time.time() - inicio
    return r


ESTRATEGIAS = {
    "A prompt": dict(restringir=False, max_intentos=1),
    "B retry": dict(restringir=False, max_intentos=MAX_INTENTOS),
    "C constrained": dict(restringir=True, max_intentos=1),
    "D constrained+retry": dict(restringir=True, max_intentos=MAX_INTENTOS),
}

RUTA_DATASET = "data/documentos.jsonl"
RUTA_RESULTADOS = "resultados_comparativa.json"


def cargar_documentos() -> list[dict]:
    if os.path.exists(RUTA_DATASET):
        with open(RUTA_DATASET, encoding="utf-8") as f:
            return [json.loads(linea) for linea in f if linea.strip()]
    return [{"id": f"ej{i}", "tipo": "ejemplo", "texto": t}
            for i, t in enumerate(DOCUMENTOS_EJEMPLO, 1)]


def imprimir_tabla(titulo: str, filas: dict[str, list[dict]]) -> None:
    print(f"\n{titulo:<26}{'N':>4}{'Estricto':>9}{'Tolerante':>10}{'Final':>7}"
          f"{'Intentos':>10}{'Tokens':>8}{'Seg':>7}")
    for nombre, rs in filas.items():
        n = len(rs)
        if not n:
            continue
        print(f"{nombre:<26}{n:>4}"
              f"{sum(r['estricto_primer_intento'] for r in rs) / n:>8.0%} "
              f"{sum(r['valido_primer_intento'] for r in rs) / n:>9.0%} "
              f"{sum(r['valido'] for r in rs) / n:>6.0%} "
              f"{sum(r['intentos'] for r in rs) / n:>9.2f} "
              f"{sum(r['tokens'] for r in rs) / n:>7.0f} "
              f"{sum(r['segundos'] for r in rs) / n:>6.1f}")


def main(documentos: list[dict] | None = None) -> dict:
    documentos = documentos or cargar_documentos()
    if "--n" in sys.argv:
        documentos = documentos[:int(sys.argv[sys.argv.index("--n") + 1])]
    print(f"{len(documentos)} documentos, {len(ESTRATEGIAS)} estrategias", flush=True)

    resultados = {nombre: [] for nombre in ESTRATEGIAS}
    # Se recorre por documento y se guarda tras cada uno: si el proceso se
    # interrumpe, lo ya calculado queda en el fichero de resultados.
    for i, doc in enumerate(documentos, 1):
        for nombre, cfg in ESTRATEGIAS.items():
            r = ejecutar(doc["texto"], **cfg)
            r["id"], r["tipo"] = doc["id"], doc["tipo"]
            resultados[nombre].append(r)
            print(f"[{i}/{len(documentos)}] {doc['id']} ({doc['tipo']}) {nombre}: "
                  f"{'válido' if r['valido'] else 'NO válido'} en {r['intentos']} "
                  f"intento(s), {r['segundos']:.0f} s", flush=True)
        with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)

    imprimir_tabla("Estrategia", resultados)
    print("Estricto: salida en bruto válida al primer intento. Tolerante: válida al "
          "primer intento\ntras limpiar el envoltorio. Final: válida tras los reintentos.")

    for tipo in dict.fromkeys(d["tipo"] for d in documentos):
        imprimir_tabla(f"Tipo: {tipo}"[:25], {
            nombre: [r for r in rs if r["tipo"] == tipo]
            for nombre, rs in resultados.items()})

    print("\nTipos de error en el primer intento (modo tolerante):")
    for nombre, rs in resultados.items():
        tipos = Counter(t for r in rs for t in r["errores_primer_intento"])
        print(f"  {nombre:<22}{dict(tipos) or '-'}")
    return resultados


if __name__ == "__main__":
    main()