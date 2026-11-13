"""Genera data/train.jsonl y data/test.jsonl: documentos sintéticos CON respuesta correcta.

Uso:  python scripts/generar_dataset_v2.py
Cada línea: {"id", "tipo", "texto", "esperado", "base", "otras"}
Las entidades de test no aparecen en train, para que acertar no sea memorizar nombres.
"""
import json
import os
import random

ENT_TRAIN = [
    "Contoso S.A.", "Fabrikam Ibérica S.L.", "Northwind Traders", "Tailspin Toys",
    "Adatum Corporation", "Litware Consulting", "Proseware S.L.", "Wingtip Toys",
    "Woodgrove Bank", "Fourth Coffee", "Lucerne Publishing", "Trey Research",
    "Relecloud", "VanArsdel", "Bellows College",
]
ENT_TEST = ["Alpine Ski House", "Blue Yonder Airlines", "Coho Winery",
            "Consolidated Messenger", "Margie's Travel", "Humongous Insurance",
            "Graphic Design Institute", "Southridge Video"]
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


def generar(rnd: random.Random, entidades: list[str], n: int, prefijo: str) -> list[dict]:
    def fecha():
        d, m = rnd.randint(1, 28), rnd.randint(1, 12)
        iso = f"2026-{m:02d}-{d:02d}"
        texto = rnd.choice([f"{d} de {MESES[m - 1]} de 2026", f"{d:02d}/{m:02d}/2026",
                            f"{d}-{m}-2026", iso])
        return texto, iso

    def hallazgo(r):
        return rnd.choice(HALLAZGOS[r]).format(a=rnd.choice(AREAS))

    def limpio(e):
        r = rnd.choice(["ALTO", "MEDIO", "BAJO"])
        ft, fi = fecha()
        acc = [rnd.choice(ACCIONES)] if r != "BAJO" else []
        t = f"Informe de cumplimiento: {e} {hallazgo(r)}. Revisión realizada el {ft}." + "".join(" " + a for a in acc)
        return t, dict(entidad=e, riesgo=r, fecha_revision=fi, acciones=acc)

    def variante(e):
        base = e.split()[0]
        nombre, forma = rnd.choice([
            (f"{base.upper()}, S. A.", f"{base.upper()}, S. A."),
            (base, f"el grupo {base} (en adelante, «{base}»)"),
            (base, f"{base} y sus sociedades dependientes"),
            (f"{base}, S.L.U.", f"la mercantil {base}, S.L.U."),
        ])
        r = rnd.choice(["MEDIO", "BAJO"])
        ft, fi = fecha()
        a = rnd.choice(ACCIONES)
        t = f"Acta de revisión del {ft}. Se ha evaluado a {forma}, que {hallazgo(r)}. {a}"
        return t, dict(entidad=nombre, riesgo=r, fecha_revision=fi, acciones=[a])

    def alto_sin_acciones(e):
        ft, fi = fecha()
        t = (f"Informe urgente: {e} {hallazgo('ALTO')} y, además, {hallazgo('ALTO')}. "
             f"La situación se considera crítica. Fecha de revisión: {ft}.")
        return t, dict(entidad=e, riesgo="ALTO", fecha_revision=fi, acciones=[])

    def ruidoso(e):
        otras = rnd.sample([x for x in entidades if x != e], 2)
        r = rnd.choice(["ALTO", "MEDIO"])
        ft, fi = fecha()
        a = rnd.choice(ACCIONES)
        frases = [x.format(otra=o) for x, o in zip(rnd.sample(RUIDO, 5), otras * 3)]
        frases.insert(rnd.randint(2, 4), f"Como conclusión principal, {e} {hallazgo(r)}. {a}")
        t = f"Informe de auditoría emitido el {ft}. " + " ".join(frases)
        return t, dict(entidad=e, riesgo=r, fecha_revision=fi, acciones=[a]), otras

    def sin_fecha(e):
        r = rnd.choice(["MEDIO", "BAJO"])
        a = rnd.choice(ACCIONES)
        t = f"Nota interna sobre {e}: la entidad {hallazgo(r)}. {a}"
        return t, dict(entidad=e, riesgo=r, fecha_revision=None, acciones=[a])

    def sin_riesgo(e):
        ft, fi = fecha()
        t = (f"Nota del {ft}: se ha recibido documentación de {e} relativa a "
             f"{rnd.choice(AREAS)}. Queda pendiente de análisis por el equipo de cumplimiento.")
        return t, dict(entidad=e, riesgo=None, fecha_revision=fi, acciones=[])

    plan = [("limpio", limpio, 0.25), ("variante_entidad", variante, 0.15),
            ("alto_sin_acciones", alto_sin_acciones, 0.15), ("ruidoso", ruidoso, 0.15),
            ("sin_fecha", sin_fecha, 0.15), ("sin_riesgo", sin_riesgo, 0.15)]
    docs = []
    for tipo, fn, peso in plan:
        for _ in range(round(n * peso)):
            e = rnd.choice(entidades)
            res = fn(e)
            texto, esperado = res[0].replace(" de el ", " del ").replace(" a el ", " al "), res[1]
            otras = [o.split()[0] for o in res[2]] if len(res) > 2 else []
            docs.append({"tipo": tipo, "texto": texto, "esperado": esperado,
                         "base": e.split()[0], "otras": otras})
    rnd.shuffle(docs)
    for i, d in enumerate(docs, 1):
        d["id"] = f"{prefijo}{i:03d}"
    return docs


def guardar(ruta: str, docs: list[dict]) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(docs)} documentos en {ruta}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    guardar("data/train.jsonl", generar(random.Random(101), ENT_TRAIN, 400, "tr"))
    guardar("data/test.jsonl", generar(random.Random(202), ENT_TEST, 60, "te"))