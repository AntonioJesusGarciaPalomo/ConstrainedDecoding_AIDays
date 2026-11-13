"""Mide forma y contenido de un modelo sobre data/test.jsonl, sin y con restricción.

Forma:      ¿la salida es un JSON que cumple el contrato?
Contenido:  ¿los valores coinciden con la respuesta correcta del documento?

Uso:  python scripts/06_calidad.py RUTA_DEL_MODELO --nombre qwen-0.5b-int4
      python scripts/06_calidad.py RUTA_DEL_MODELO --nombre prueba --n 5
      Opciones: --datos data/test_fuera.jsonl   --max-tokens 400
"""
import argparse
import json
import re
import time
import unicodedata

from pydantic import ValidationError

from contrato_v2 import SCHEMA_JSON, EvaluacionV2, mensajes

MAX_TOKENS_NUEVOS = 220
_modelo = _tokenizer = None


def generar(ruta: str, texto: str, restringir: bool) -> tuple[str, int, float]:
    """Devuelve (salida, tokens generados, segundos)."""
    global _modelo, _tokenizer
    import onnxruntime_genai as og

    if _modelo is None:
        _modelo = og.Model(ruta)
        _tokenizer = og.Tokenizer(_modelo)
    prompt = _tokenizer.apply_chat_template(json.dumps(mensajes(texto)), add_generation_prompt=True)
    entrada = _tokenizer.encode(prompt)
    params = og.GeneratorParams(_modelo)
    params.set_search_options(max_length=len(entrada) + MAX_TOKENS_NUEVOS)
    if restringir:
        params.set_guidance("json_schema", SCHEMA_JSON)
    generador = og.Generator(_modelo, params)
    generador.append_tokens(entrada)
    stream = _tokenizer.create_stream()
    salida, n, inicio = "", 0, time.time()
    while not generador.is_done():
        generador.generate_next_token()
        salida += stream.decode(generador.get_next_tokens()[0])
        n += 1
    return salida, n, time.time() - inicio


def extraer_json(texto: str) -> str:
    i, j = texto.find("{"), texto.rfind("}")
    return texto[i:j + 1] if i != -1 and j > i else texto


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def evaluar(salida: str, doc: dict) -> dict:
    r = {"estricto": False, "tolerante": False, "entidad": False, "riesgo": False,
         "fecha": False, "acciones": False, "todo": False}
    try:
        EvaluacionV2.model_validate_json(salida)
        r["estricto"] = True
    except ValidationError:
        pass
    try:
        p = EvaluacionV2.model_validate_json(extraer_json(salida))
    except ValidationError:
        return r                      # sin forma válida, el contenido cuenta como fallo
    r["tolerante"] = True
    e = doc["esperado"]
    pred = norm(p.entidad)
    r["entidad"] = norm(doc["base"]) in pred and not any(norm(o) in pred for o in doc["otras"])
    r["riesgo"] = p.riesgo == e["riesgo"]
    r["fecha"] = (p.fecha_revision.isoformat() if p.fecha_revision else None) == e["fecha_revision"]
    r["acciones"] = (len(p.acciones) == 0) == (len(e["acciones"]) == 0)
    r["todo"] = all(r[k] for k in ("entidad", "riesgo", "fecha", "acciones"))
    return r


def main() -> None:
    global MAX_TOKENS_NUEVOS
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--datos", default="data/test.jsonl")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_NUEVOS,
                    help="límite de tokens generados por respuesta")
    a = ap.parse_args()
    MAX_TOKENS_NUEVOS = a.max_tokens

    with open(a.datos, encoding="utf-8") as f:
        docs = [json.loads(linea) for linea in f if linea.strip()]
    if a.n:
        docs = docs[:a.n]
    print(f"Modelo: {a.modelo} | {len(docs)} documentos", flush=True)

    resultados = {"Sin restricción": [], "Con restricción": []}
    for i, doc in enumerate(docs, 1):
        for nombre, restringir in (("Sin restricción", False), ("Con restricción", True)):
            salida, tokens, seg = generar(a.modelo, doc["texto"], restringir)
            r = evaluar(salida, doc)
            r.update(id=doc["id"], tipo=doc["tipo"], salida=salida, tokens=tokens, segundos=seg)
            resultados[nombre].append(r)
        s, c = resultados["Sin restricción"][-1], resultados["Con restricción"][-1]
        print(f"[{i}/{len(docs)}] {doc['id']} ({doc['tipo']}): sin restricción "
              f"{'OK' if s['todo'] else 'x'} | con restricción {'OK' if c['todo'] else 'x'}", flush=True)
        with open(f"data/calidad_{a.nombre}.json", "w", encoding="utf-8") as f:
            json.dump({"modelo": a.modelo, "resultados": resultados}, f, ensure_ascii=False, indent=1)

    n = len(docs)
    print(f"\n== {a.nombre} ({n} documentos)")
    print(f"{'':<17}{'FORMA':^19}{'CONTENIDO CORRECTO':^47}")
    print(f"{'Estrategia':<17}{'Tal cual':>9}{'Limpia':>9}{'Entidad':>10}{'Riesgo':>8}"
          f"{'Fecha':>8}{'Acciones':>10}{'Todo':>8}{'Tokens':>8}{'Seg':>7}")
    for nombre, rs in resultados.items():
        pct = lambda k: f"{sum(r[k] for r in rs) / n:.0%}"
        print(f"{nombre:<17}{pct('estricto'):>9}{pct('tolerante'):>9}{pct('entidad'):>10}"
              f"{pct('riesgo'):>8}{pct('fecha'):>8}{pct('acciones'):>10}{pct('todo'):>8}"
              f"{sum(r['tokens'] for r in rs) / n:>8.0f}{sum(r['segundos'] for r in rs) / n:>7.1f}")
    print("Tal cual: salida en bruto válida. Limpia: válida tras quitar el envoltorio.\n"
          "Contenido: sobre todos los documentos; una salida sin forma válida cuenta como fallo.\n"
          "Acciones: acierta si hay acciones solo cuando el documento las indica.")


if __name__ == "__main__":
    main()