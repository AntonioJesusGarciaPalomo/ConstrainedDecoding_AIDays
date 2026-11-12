"""Genera data/documentos.jsonl: dataset sintético y reproducible (semilla fija).

Uso:  python scripts/generar_dataset.py
Cada línea: {"id", "tipo", "texto", "entidad", "tiene_fecha"}
"""
import json
import os
import random

rnd = random.Random(13)

ENTIDADES = [
    "Contoso S.A.", "Fabrikam Ibérica S.L.", "Northwind Traders", "Tailspin Toys",
    "Adatum Corporation", "Litware Consulting", "Proseware S.L.", "Wingtip Toys",
    "Woodgrove Bank", "Fourth Coffee", "Lucerne Publishing", "Trey Research",
    "Relecloud", "VanArsdel", "Alpine Ski House", "Bellows College",
    "Blue Yonder Airlines", "Coho Winery", "Consolidated Messenger", "Margie's Travel",
]
AREAS = [
    "el control de accesos", "la gestión de proveedores críticos",
    "la protección de datos personales", "la continuidad de negocio",
    "la prevención del blanqueo de capitales", "la gestión de incidentes de seguridad",
    "el registro de actividades de tratamiento", "la segregación de funciones",
]
HALLAZGOS = {
    "ALTO": ["presenta deficiencias graves en {a}", "incumple de forma reiterada los "
             "requisitos de {a}", "carece de controles efectivos sobre {a}"],
    "MEDIO": ["muestra retrasos puntuales en {a}", "tiene documentación incompleta "
              "sobre {a}", "aplica de forma desigual los controles de {a}"],
    "BAJO": ["no presenta incidencias relevantes en {a}", "mantiene controles adecuados "
             "sobre {a}", "cumple los requisitos de {a} con observaciones menores"],
}
ACCIONES = [
    "Se exige rotar las credenciales privilegiadas en 30 días.",
    "Se requiere un plan de remediación aprobado por la dirección.",
    "Se recomienda revisar el procedimiento y formar al personal.",
    "Debe designarse un responsable y fijar revisiones trimestrales.",
]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
RUIDO = [
    "El equipo auditor agradece la colaboración del personal entrevistado.",
    "La reunión de cierre se celebró en las oficinas centrales con la dirección.",
    "Este informe sustituye a la versión preliminar distribuida anteriormente.",
    "Se revisaron 42 expedientes seleccionados por muestreo aleatorio.",
    "El alcance no incluyó a las filiales de fuera de la Unión Europea.",
    "En la auditoría anterior, realizada a {otra}, no se detectaron hallazgos.",
    "El proveedor {otra} facilitó parte de la evidencia documental.",
]


def fecha_larga():
    d, m = rnd.randint(1, 28), rnd.randint(1, 12)
    return f"{d} de {MESES[m - 1]} de 2026"


def hallazgo(riesgo):
    return rnd.choice(HALLAZGOS[riesgo]).format(a=rnd.choice(AREAS))


def limpio(e):
    r = rnd.choice(["ALTO", "MEDIO", "BAJO"])
    accion = " " + rnd.choice(ACCIONES) if r != "BAJO" else ""
    return f"Informe de cumplimiento: {e} {hallazgo(r)}. Revisión realizada el {fecha_larga()}.{accion}"


def variante_entidad(e):
    base = e.split()[0]
    forma = rnd.choice([
        f"{base.upper()}, S. A.", f"el grupo {base} (en adelante, «{base}»)",
        f"{base} y sus sociedades dependientes", f"la mercantil {base}, S.L.U.",
    ])
    r = rnd.choice(["MEDIO", "BAJO"])
    return f"Acta de revisión del {fecha_larga()}. Se ha evaluado a {forma}, que {hallazgo(r)}. {rnd.choice(ACCIONES)}"


def alto_sin_acciones(e):
    return (f"Informe urgente: {e} {hallazgo('ALTO')} y, además, {hallazgo('ALTO')}. "
            f"La situación se considera crítica. Fecha de revisión: {fecha_larga()}.")


def fecha_variada(e):
    d, m = rnd.randint(1, 12), rnd.randint(1, 12)
    f = rnd.choice([f"{d:02d}/{m:02d}/2026", f"{MESES[m - 1]} de 2026",
                    f"el {d}-{m}-26", f"la primera quincena de {MESES[m - 1]} de 2026"])
    r = rnd.choice(["MEDIO", "BAJO"])
    return f"Auditoría de {e} ({f}): la entidad {hallazgo(r)}. {rnd.choice(ACCIONES)}"


def largo_ruidoso(e):
    otras = rnd.sample([x for x in ENTIDADES if x != e], 2)
    r = rnd.choice(["ALTO", "MEDIO"])
    frases = [x.format(otra=o) for x, o in zip(rnd.sample(RUIDO, 5), otras * 3)]
    nucleo = f"Como conclusión principal, {e} {hallazgo(r)}. {rnd.choice(ACCIONES)}"
    frases.insert(rnd.randint(2, 4), nucleo)
    return f"Informe de auditoría emitido el {fecha_larga()}. " + " ".join(frases)


def incompleto(e):
    if rnd.random() < 0.5:   # sin fecha
        r = rnd.choice(["MEDIO", "BAJO"])
        return f"Nota interna sobre {e}: la entidad {hallazgo(r)}. {rnd.choice(ACCIONES)}"
    return (f"Nota del {fecha_larga()}: se ha recibido documentación de {e} relativa a "
            f"{rnd.choice(AREAS)}. Queda pendiente de análisis por el equipo de cumplimiento.")


PLAN = [("limpio", limpio, 15), ("variante_entidad", variante_entidad, 10),
        ("alto_sin_acciones", alto_sin_acciones, 8), ("fecha_variada", fecha_variada, 7),
        ("largo_ruidoso", largo_ruidoso, 5), ("incompleto", incompleto, 5)]

docs = []
for tipo, fn, n in PLAN:
    for _ in range(n):
        e = rnd.choice(ENTIDADES)
        texto = fn(e).replace(" de el ", " del ")
        docs.append({"tipo": tipo, "texto": texto, "entidad": e,
                     "tiene_fecha": not (tipo == "incompleto" and "Nota interna" in texto)})
rnd.shuffle(docs)   # mezclados: una ejecución parcial ya cubre todos los tipos
for i, d in enumerate(docs, 1):
    d["id"] = f"doc{i:02d}"

os.makedirs("data", exist_ok=True)
with open("data/documentos.jsonl", "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps({k: d[k] for k in ("id", "tipo", "texto", "entidad", "tiene_fecha")},
                           ensure_ascii=False) + "\n")
print(f"{len(docs)} documentos en data/documentos.jsonl")