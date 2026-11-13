# Constrained decoding: que el modelo cumpla el contrato

Material del workshop **«Constrained decoding con modelos locales: optimizar, imponer, afinar»**, impartido en línea por Teams.

Le pides a un modelo de lenguaje un JSON con cuatro campos. Te devuelve esto:

````text
```json
{
  "entidad": "Contoso S.A.",
  "risgo": "BAJO",
  "puntuacion": 95,
  "fecha_revision": "2026-03-03"
}
```

Explanation:
- **Entidad**: The entity is "Contoso S.A."
...
````

Casi. Pero tu aplicación hace `json.loads(salida)` y se rompe: hay un bloque markdown, una explicación en inglés que nadie pidió y un campo que se llama `risgo`. Y aunque lo arreglaras todo a mano, el documento hablaba de «deficiencias graves» y el modelo ha dicho riesgo BAJO.

En este workshop vamos a resolver las dos cosas, por separado y midiendo cada paso, en tu portátil y sin GPU.

## La idea en tres verbos

| Paso | Qué hacemos | Con qué | Qué resuelve |
| --- | --- | --- | --- |
| **Optimizar** | Convertir un modelo a ONNX y cuantizarlo a 4 bits | Microsoft Olive | Que el modelo quepa y corra en un portátil |
| **Imponer** | Obligar a que la salida cumpla un JSON Schema | ONNX Runtime GenAI + LLGuidance | La **forma** de la respuesta |
| **Afinar** | Enseñar al modelo nuestro criterio con un adaptador LoRA | PEFT + Olive | El **contenido** de la respuesta |

La frase que resume la sesión: **la restricción garantiza la forma, no la verdad**. Para la verdad hacen falta otras capas, y aquí veremos una.

## Antes de la sesión: prepara tu entorno (15 a 20 minutos)

Hazlo antes de conectarte. En las dos horas del workshop no habrá tiempo para instalar nada.

### Qué necesitas

- Un portátil con unos 8 GB de RAM y unos 8 GB de disco libres. No hace falta GPU.
- Linux, macOS, o Windows con WSL 2 y una distribución Ubuntu.
- `git` y conexión a internet.

Trabajaremos con **Qwen2.5 de 0,5 mil millones de parámetros**: un modelo abierto, pequeño y rápido. En un portátil normal genera unos 75 tokens por segundo, así que los experimentos tardan segundos y no minutos.

### Paso 1. Clona el repositorio

En Windows, hazlo **dentro de WSL** (en `/home/...`), no en `C:\`.

```bash
cd ~
git clone https://github.com/AntonioJesusGarciaPalomo/ConstrainedDecoding_AIDays.git
cd ConstrainedDecoding_AIDays
```

### Paso 2. Crea el entorno de Python

Usamos [uv](https://docs.astral.sh/uv/), que instala su propio Python 3.12 sin tocar el de tu sistema.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

source .venv/bin/activate
```

En macOS, sustituye la línea de `torch` por `uv pip install --python .venv/bin/python torch`.

> En cada terminal nueva tendrás que repetir `source .venv/bin/activate` desde la carpeta del proyecto.

### Paso 3. Convierte tu primer modelo con Olive

Este comando ya es el primer paso del workshop. Descarga el modelo original (1 GB), lo convierte a ONNX y lo cuantiza a 4 bits. Tarda alrededor de un minuto.

```bash
olive run-pass --pass-name ModelBuilder \
  --pass-config '{"precision": "int4", "extra_options": {"hf_token": "false"}}' \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu --provider CPUExecutionProvider \
  --output_path models/qwen-0.5b-int4
```

Verás pasar las capas del modelo («Reading layer 0… 23») y, al final, `Model is saved at .../models/qwen-0.5b-int4`.

### Paso 4. Comprueba que todo funciona

```bash
python scripts/05_prueba_modelo.py models/qwen-0.5b-int4
```

Deberías ver dos bloques parecidos a estos:

```text
== Sin restricción (128 tokens, 75.3 tokens/s)
(un JSON envuelto en markdown, con texto alrededor)
-> NO válido: JSONDecodeError ...

== Con restricción (50 tokens, 80.6 tokens/s)
{
  "entidad": "Contoso S.A.",
  "riesgo": "BAJO",
  "puntuacion": 50,
  "fecha_revision": "2026-03-03"
}
-> Válido contra el esquema
```

Si el segundo bloque termina en **«Válido contra el esquema»**, estás listo. El texto exacto puede variar un poco de un equipo a otro.

### El día de la sesión

Diez minutos antes, abre una terminal en la carpeta del proyecto, activa el entorno y repite el paso 4. Si tienes dos pantallas, usa una para Teams y otra para la terminal.

### Si no consigues montarlo

Conéctate igualmente. Todo se mostrará en pantalla compartida y los resultados de cada experimento están en la carpeta `data/`. Si puedes, avisa antes con el error que te aparece.

## El recorrido del workshop

Cada bloque tiene la misma estructura: qué hacemos, qué comando ejecutas y qué deberías aprender de lo que ves.

### 1. Optimizar: del modelo original a tu portátil

El modelo que publica su autor está pensado para entrenar en GPU con PyTorch. Para ejecutarlo en una CPU hay que hacer dos cosas: **convertirlo** a ONNX, un formato que no depende de PyTorch, y **cuantizarlo**, es decir, guardar cada peso con 4 bits en lugar de 32.

Ya lo hiciste en el paso 3. Ahora convierte el mismo modelo **sin cuantizar** y compara:

```bash
olive run-pass --pass-name ModelBuilder \
  --pass-config '{"precision": "fp32", "extra_options": {"hf_token": "false"}}' \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu --provider CPUExecutionProvider \
  --output_path models/qwen-0.5b-fp32

du -sh models/qwen-0.5b-int4 models/qwen-0.5b-fp32
python scripts/05_prueba_modelo.py models/qwen-0.5b-fp32
```

**Qué mirar:** el tamaño en disco y los tokens por segundo. En nuestras medidas, el modelo de 4 bits fue 3,6 veces más rápido.

**Un detalle sobre Olive.** Olive no convierte por sí mismo: es un orquestador que encadena pasos, a los que llama *passes*. El pass `ModelBuilder` usa por dentro el conversor de ONNX Runtime GenAI. Para un caso sencillo como este, Olive es una capa fina. Compensa cuando hay varios dispositivos de destino, varias formas de cuantizar o hay que medir cuál es mejor.

### 2. El problema: un prompt no es un contrato

Vuelve a mirar el bloque «Sin restricción» del paso 4. El prompt pedía un JSON con cuatro campos concretos. El modelo, por su cuenta:

- envuelve la respuesta en markdown y añade explicaciones;
- a veces cambia o escribe mal el nombre de un campo;
- y nada de eso depende de ti.

Pedir una estructura por texto es una sugerencia. Una aplicación necesita un contrato.

### 3. Imponer: constrained decoding

Un modelo genera texto token a token: en cada paso calcula una probabilidad para cada token de su vocabulario y elige uno. **Constrained decoding** añade un filtro en ese punto: una gramática, derivada de tu JSON Schema, marca qué tokens mantienen la salida dentro del contrato, y el resto no pueden salir.

No se corrige la respuesta después. Se impide que el token inválido llegue a escribirse.

En código es una línea. Ábrela en `scripts/05_prueba_modelo.py`:

```python
params.set_guidance("json_schema", json.dumps(schema))
```

**Qué mirar:** el bloque «Con restricción» produce un JSON limpio y, además, con menos tokens. El modelo deja de gastar palabras en introducciones.

### 4. Medir: forma y contenido, por separado

Una prueba con un documento es una anécdota. Vamos a medir sobre 60 documentos sintéticos de cumplimiento normativo, cada uno con su respuesta correcta conocida:

```bash
python scripts/06_calidad.py models/qwen-0.5b-int4 --nombre mi-qwen
python scripts/08_desglose.py mi-qwen
```

Tarda unos dos minutos. La tabla separa dos preguntas distintas:

- **Forma:** ¿la salida es un JSON que cumple el contrato? «Tal cual» es la salida en bruto; «Limpia», después de quitarle el envoltorio.
- **Contenido:** ¿los valores son los correctos? Se mide campo a campo y en conjunto («Todo»).

Esto es lo que obtuvimos nosotros:

| Qwen 0,5 en 4 bits | Forma tal cual | Riesgo | Fecha | Todo correcto |
| --- | --- | --- | --- | --- |
| Sin restricción | 72 % | 15 % | 58 % | 10 % |
| Con restricción | 100 % | 23 % | 73 % | 7 % |

**Qué aprender:** la restricción lleva la forma al 100 % y no estropea el contenido. Pero el contenido sigue siendo malo: con forma perfecta, el modelo acierta todo en 4 documentos de 60.

### 5. El límite: forma, no verdad

El desglose del paso anterior enseña dónde falla. Fíjate en estos tipos de documento:

- **`sin_fecha`:** el documento no dice cuándo fue la revisión. El modelo inventa una fecha con formato impecable.
- **`alto_sin_acciones` y `sin_riesgo`:** el documento no propone ninguna acción. El modelo rellena la lista igualmente.

El contrato permite `null` y lista vacía, y el prompt lo pide de forma expresa. No basta: el modelo prefiere inventar antes que decir «no consta». Y todas esas salidas son **válidas contra el esquema**.

Lo vimos también en modelos mayores. Phi-3.5, casi ocho veces más grande, inventó acciones en 18 de 18 documentos que no contenían ninguna.

### 6. Afinar: enseñar al modelo nuestro criterio

La forma ya está resuelta. Para el contenido, entrenamos un **adaptador LoRA**: en lugar de reentrenar el modelo entero, se ajusta un pequeño conjunto de pesos añadidos, aquí el 1,7 % del total. Lo entrenamos con 400 documentos sintéticos y su respuesta correcta.

El resultado, medido con el mismo test de 60 documentos y con restricción:

| Modelo | Todo correcto | Segundos por documento |
| --- | --- | --- |
| Qwen 0,5 sin afinar | 7 % | 1,0 |
| Qwen 1,5 sin afinar | 22 % | 2,1 |
| Phi-3.5 sin afinar | 30 % | 7,0 |
| **Qwen 0,5 con LoRA** | **100 %** | **0,7** |

¿Se ha aprendido el examen de memoria? Lo comprobamos con 14 documentos escritos a mano, con otro formato y otro vocabulario. Pasó de acertar 4 a acertar 12, por encima de los 8 de Phi-3.5 sin afinar y tardando la décima parte. Y aprendió lo que más nos importaba: a responder `null` cuando el dato no está.

El entrenamiento completo tarda unos 90 minutos en CPU, así que en la sesión veremos los resultados ya calculados. Puedes comprobar en un minuto que funciona en tu equipo:

```bash
python scripts/07_lora.py --prueba
```

Y, si quieres reproducirlo entero en casa:

```bash
python scripts/07_lora.py
olive run-pass --pass-name ModelBuilder \
  --pass-config '{"precision": "int4", "extra_options": {"hf_token": "false"}}' \
  --model_name_or_path models/qwen-0.5b-lora \
  --device cpu --provider CPUExecutionProvider \
  --output_path models/qwen-0.5b-lora-int4
python scripts/06_calidad.py models/qwen-0.5b-lora-int4 --nombre mi-lora
```

Fíjate en el segundo comando: Olive vuelve a aparecer. La misma herramienta que preparó el modelo original empaqueta el modelo afinado.

**Qué aprender:** afinar no sustituye a la restricción. El modelo afinado, sin restricción, sigue fallando la forma en el 15 % de los casos. Con las dos capas, llega al 100 %.

### 7. Tu turno: diseña un contrato

Abre `scripts/05_prueba_modelo.py`. Arriba están el `schema` y el `documento`. Cámbialos y ejecuta; cada prueba tarda dos segundos.

Ideas para experimentar:

1. Añade un campo con una lista cerrada de valores (`enum`).
2. Permite que la fecha sea nula: `{"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]}`. Luego borra la fecha del documento. ¿Usa el modelo esa salida?
3. Quita `"additionalProperties": False`. ¿Aparece algún campo que no pediste?
4. Añade `"uniqueItems": true` a una lista. ¿Qué ocurre?
5. Escribe mal una palabra clave, por ejemplo `minLenght`. ¿Avisa alguien?

Comparte en el chat un caso en el que la salida sea válida contra tu esquema y, aun así, falsa.

## Seis reglas para diseñar un contrato

Cada una sale de un fallo que nos encontramos al preparar este material.

1. **Da una salida a «no consta».** Un campo obligatorio sin opción de nulo fabrica datos. Es necesario, aunque ya has visto que no es suficiente.
2. **Cierra el objeto.** Sin `additionalProperties: false`, el modelo puede añadir campos por su cuenta.
3. **Pon los límites en el esquema, no en la descripción.** Un rango escrito en un comentario no se impone.
4. **Enumera los valores cerrados.** Con un texto libre saldrá «Alto», «alto» y «ALTO».
5. **Deja margen de tokens.** Si la respuesta alcanza el límite, el JSON queda cortado e inválido, con restricción o sin ella.
6. **Revisa las palabras clave.** Las no soportadas dan error; las mal escritas se ignoran sin aviso.

## Resultados de referencia

Están en `data/`, para que puedas compararlos con los tuyos o seguir la sesión sin ejecutar nada.

**Los cinco modelos, con restricción, sobre 60 documentos:**

| Modelo | Forma | Riesgo | Fecha | Acciones | Todo correcto |
| --- | --- | --- | --- | --- | --- |
| Qwen 0,5 sin cuantizar | 100 % | 23 % | 48 % | 62 % | 5 % |
| Qwen 0,5 en 4 bits | 100 % | 23 % | 73 % | 63 % | 7 % |
| Qwen 1,5 en 4 bits | 100 % | 35 % | 77 % | 85 % | 22 % |
| Phi-3.5 en 4 bits | 100 % | 48 % | 80 % | 70 % | 30 % |
| Qwen 0,5 con LoRA, en 4 bits | 100 % | 100 % | 100 % | 100 % | 100 % |

**Fuera de plantilla, con restricción, sobre 14 documentos escritos a mano:**

| Modelo | Riesgo | Fecha | Acciones | Todo correcto |
| --- | --- | --- | --- | --- |
| Qwen 0,5 sin afinar | 36 % | 79 % | 71 % | 4 de 14 |
| Qwen 1,5 sin afinar | 21 % | 64 % | 100 % | 3 de 14 |
| Phi-3.5 sin afinar | 93 % | 79 % | 79 % | 8 de 14 |
| Qwen 0,5 con LoRA | 93 % | 100 % | 93 % | 12 de 14 |

Medido en un portátil con Intel Core Ultra 7 155U, solo CPU.

### Lo que estos números no demuestran

- Los documentos son **sintéticos**. El test de 60 sale del mismo generador que los datos de entrenamiento, y los 14 «fuera de plantilla» los escribió la misma persona. Miden que el mecanismo funciona, no que el modelo sirva para tus documentos.
- Con 14 documentos, cada uno vale 7 puntos. La diferencia entre 4 y 12 es clara; el porcentaje exacto, no.
- Las etiquetas de riesgo son **nuestro criterio**. Lo que el LoRA aprende es esa convención, que es justo para lo que sirve afinar.
- Es un solo entrenamiento, con una sola semilla.

## Para ir más lejos: Phi-3.5

Si tienes 16 GB de RAM y quieres repetir los experimentos con un modelo mayor, este script descarga Phi-3.5 mini ya convertido (2,8 GB) y lo prueba:

```bash
bash scripts/00_preparar_modelo.sh
export MODEL_DIR=$PWD/models/phi-3.5-cpu/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4
python scripts/02_constrained.py
python scripts/04_comparativa.py --n 3
```

Genera a unos 10 tokens por segundo, así que la comparativa completa de 50 documentos tarda más de una hora. Sus resultados están en `data/resultados_50.json`.

## Mapa del repositorio

| Ruta | Qué es |
| --- | --- |
| `scripts/05_prueba_modelo.py` | Prueba cualquier modelo, sin y con restricción. El punto de partida |
| `scripts/06_calidad.py` | Mide forma y contenido sobre un conjunto de documentos |
| `scripts/08_desglose.py` | Desglosa los resultados por tipo de documento y por riesgo |
| `scripts/07_lora.py` | Entrena el adaptador LoRA en CPU |
| `scripts/contrato_v2.py` | El contrato Pydantic que comparten evaluación y entrenamiento |
| `scripts/generar_dataset_v2.py` | Genera los datos de entrenamiento y de test, con semilla fija |
| `scripts/00_` a `04_` | Los experimentos con Phi-3.5 |
| `data/train.jsonl`, `data/test.jsonl` | 400 y 60 documentos sintéticos con su respuesta correcta |
| `data/test_fuera.jsonl` | 14 documentos escritos a mano |
| `data/calidad_*.json` | Resultados de referencia de cada modelo |

## Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| `olive: command not found` o `No module named ...` | El entorno no está activo. Ejecuta `source .venv/bin/activate` desde la carpeta del proyecto |
| El proceso termina con `Killed` | Falta de memoria. Cierra otros programas. En Windows, sube la memoria de WSL en `%USERPROFILE%\.wslconfig` y ejecuta `wsl --shutdown` |
| Error de certificado SSL al instalar o descargar | Tu red corporativa inspecciona el tráfico. Prueba sin la VPN o desde otra red |
| Todo va muy lento en Windows | El proyecto está en `C:\`. Clónalo dentro de WSL |
| `Token is required` al convertir | Falta `"hf_token": "false"` en `--pass-config` |
| `No module named 'onnxruntime_genai.models.quantized_model'` | Tienes otra versión de onnxruntime-genai. Reinstala con `requirements.txt` |

## Versiones, y por qué están fijadas

```text
olive-ai 0.13.0 · onnxruntime-genai 0.14.1 · guidance 0.3.1 · peft · Python 3.12
```

No es la versión más reciente de onnxruntime-genai, y es a propósito. Olive 0.13.0 depende de una parte interna de onnxruntime-genai que se reorganizó a partir de la versión 0.16, así que con versiones posteriores la conversión falla. Comprobamos que los resultados de constrained decoding son idénticos en la 0.14.1 y en la 0.19.0.

Es una lección práctica del propio workshop: las herramientas de conversión y de inferencia evolucionan a ritmos distintos, y en un proyecto real las versiones se fijan.