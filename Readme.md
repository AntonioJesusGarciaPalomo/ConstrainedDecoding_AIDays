# Constrained decoding con Phi, ONNX Runtime y Guidance

Material del workshop, que se imparte en línea por Teams. Ejecutaremos un modelo Phi-3.5 en local, solo con CPU, y compararemos cuatro formas de obtener una salida estructurada: prompt, validación con reintento, constrained decoding y la combinación de ambas.

**Prepara el entorno antes de la sesión.** Lleva entre 20 y 30 minutos y descarga unos 6 GB. En las dos horas del workshop no habrá tiempo para hacerlo.

## Requisitos

- Portátil con 16 GB de RAM o más y 15 GB de disco libres. No hace falta GPU.
- Docker Desktop o Rancher Desktop (con el motor `dockerd`), en marcha.
- Visual Studio Code con la extensión **Dev Containers**.
- En Windows: WSL 2 con una distribución Ubuntu, y la extensión **WSL** de VS Code.

Probado en Windows 11 con WSL 2 y Rancher Desktop, en un equipo x64. En macOS y Linux debería funcionar igual, pero no está probado.

## Preparación

### 1. Clona el repositorio

En Windows, hazlo **dentro de WSL**, no en `C:\`. Si el proyecto está en el disco de Windows, el modelo tarda mucho más en cargar.

```bash
cd ~
git clone https://github.com/AntonioJesusGarciaPalomo/ConstrainedDecoding_AIDays.git
cd ConstrainedDecoding_AIDays
code .
```

### 2. Abre el proyecto en el contenedor

En VS Code, pulsa `F1` y elige **Dev Containers: Reopen in Container**. La primera vez tarda unos minutos, porque construye la imagen e instala los paquetes.

Sabrás que ha terminado cuando la esquina inferior izquierda diga `Dev Container: workshop-constrained`.

### 3. Descarga el modelo y comprueba que funciona

Abre una terminal en VS Code (`Terminal > New Terminal`) y ejecuta:

```bash
bash scripts/00_preparar_modelo.sh
```

El script descarga el modelo (2,8 GB), ajusta su configuración y genera un texto de prueba.

### 4. Comprueba el resultado

Si todo ha ido bien, las últimas líneas serán un texto en español y una medida de velocidad, parecida a esta:

```
86 tokens, 10.6 tokens/s
```

Después ejecuta la prueba de constrained decoding:

```bash
python scripts/02_constrained.py
```

Debe imprimir un objeto JSON y la línea `-> Válido contra el esquema`. Con eso estás listo.

## Alternativa sin Docker

Si Docker te da problemas, puedes montar el mismo entorno directamente en Linux, en WSL o en macOS con [uv](https://docs.astral.sh/uv/). Probado en Ubuntu sobre WSL 2.

```bash
cd ~/ConstrainedDecoding_AIDays
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

source .venv/bin/activate
bash scripts/00_preparar_modelo.sh
python scripts/02_constrained.py
```

En cada terminal nueva, activa el entorno y define la ruta del modelo:

```bash
source .venv/bin/activate
export MODEL_DIR=$PWD/models/phi-3.5-cpu/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4
```

## Contenido

| Ruta | Qué es |
| --- | --- |
| `scripts/01_base.py` | Inferencia básica con ONNX Runtime GenAI |
| `scripts/02_constrained.py` | Constrained decoding con un JSON Schema. Con `--libre` genera sin restricción |
| `scripts/03_guidance.py` | La misma idea con la librería Guidance |
| `scripts/04_comparativa.py` | Comparativa de las cuatro estrategias. Con `--n 3` procesa solo tres documentos |
| `scripts/generar_dataset.py` | Genera el dataset sintético, con semilla fija |
| `data/documentos.jsonl` | 50 documentos sintéticos de cumplimiento normativo |
| `data/resultados_50.json` | Resultados precalculados de la comparativa completa |

La comparativa completa tarda más de una hora en un portátil. Por eso los resultados van incluidos; en la sesión ejecutaremos solo unos pocos documentos.

## Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| El proceso termina con `Killed` | Falta de memoria. En Windows, sube la memoria de WSL en `%USERPROFILE%\.wslconfig` (por ejemplo, `memory=12GB` bajo `[wsl2]`) y ejecuta `wsl --shutdown`. Cierra otros programas pesados |
| `KeyError: 'MODEL_DIR'` | Estás fuera del contenedor, o en una terminal abierta antes de construirlo. Abre una terminal nueva dentro de VS Code |
| `hf: command not found` | Estás fuera del contenedor. Mira que la esquina inferior izquierda de VS Code diga `Dev Container` |
| Error de certificado SSL al construir | Tu red corporativa inspecciona el tráfico. Prueba sin la VPN de la empresa o desde otra red |
| El modelo tarda minutos en cargar | El proyecto está en `C:\`. Clónalo dentro de WSL, como indica el paso 1 |
| `docker` no responde | Docker Desktop o Rancher Desktop no está arrancado |

## El día de la sesión

Diez minutos antes de empezar, abre VS Code, entra en el contenedor y ejecuta `python scripts/01_base.py`. Así compruebas que todo sigue funcionando y llegas con el entorno listo.

Si tienes dos pantallas, usa una para Teams y otra para VS Code.

## Si no consigues montarlo

Podrás seguir la sesión igualmente: todo se mostrará en pantalla compartida y los resultados precalculados están en el repositorio. Si puedes, avisa antes con el error que te aparece, para intentar resolverlo antes del workshop.

## Versiones

Las versiones están fijadas en `requirements.txt`: onnxruntime-genai 0.19.0, guidance 0.3.1 y olive-ai 0.13.0, sobre Python 3.12. Modelo: `microsoft/Phi-3.5-mini-instruct-onnx`, variante CPU int4.