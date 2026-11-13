"""Desglosa los resultados de 06_calidad.py por tipo de documento y por valor de riesgo.

Uso:  python scripts/08_desglose.py qwen-0.5b-int4 qwen-0.5b-lora-int4 phi-3.5-int4
"""
import collections
import json
import sys

docs = {}
for ruta in ("data/test.jsonl", "data/test_fuera.jsonl"):
    try:
        with open(ruta, encoding="utf-8") as f:
            docs.update({d["id"]: d for d in map(json.loads, f)})
    except FileNotFoundError:
        pass

for nombre in sys.argv[1:]:
    try:
        rs = json.load(open(f"data/calidad_{nombre}.json"))["resultados"]["Con restricción"]
    except FileNotFoundError:
        print(f"\n== {nombre}: no existe data/calidad_{nombre}.json")
        continue
    print(f"\n== {nombre}, con restricción ({len(rs)} documentos)")
    print(f"{'Tipo':<20}{'N':>4}{'Entidad':>9}{'Riesgo':>8}{'Fecha':>7}{'Acciones':>10}{'Todo':>7}")
    por_tipo = collections.defaultdict(list)
    for r in rs:
        por_tipo[r["tipo"]].append(r)
    for tipo, g in por_tipo.items():
        pct = lambda k: f"{sum(r[k] for r in g) / len(g):.0%}"
        print(f"{tipo:<20}{len(g):>4}{pct('entidad'):>9}{pct('riesgo'):>8}{pct('fecha'):>7}"
              f"{pct('acciones'):>10}{pct('todo'):>7}")
    cruce = collections.Counter()
    for r in rs:
        try:
            dicho = json.loads(r["salida"]).get("riesgo")
        except Exception:
            dicho = "?"
        cruce[(docs[r["id"]]["esperado"]["riesgo"], dicho)] += 1
    valores = ["ALTO", "MEDIO", "BAJO", None]
    print("Riesgo correcto (filas) frente a riesgo respondido (columnas):")
    print(f"{'':<10}" + "".join(f"{str(v):>7}" for v in valores))
    for c in valores:
        print(f"{str(c):<10}" + "".join(f"{cruce[(c, v)]:>7}" for v in valores))