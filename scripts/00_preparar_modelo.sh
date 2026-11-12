#!/usr/bin/env bash
# Descarga el modelo, ajusta su configuración y hace una prueba de generación.
# Uso (dentro del Dev Container, desde la raíz del proyecto):
#   bash scripts/00_preparar_modelo.sh
set -euo pipefail

VARIANTE="cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
: "${MODEL_DIR:=$PWD/models/phi-3.5-cpu/$VARIANTE}"
export MODEL_DIR

echo "== 1/3 Descargando el modelo (2,8 GB) en models/phi-3.5-cpu"
hf download microsoft/Phi-3.5-mini-instruct-onnx \
  --include "cpu_and_mobile/*" \
  --local-dir ./models/phi-3.5-cpu

if [ ! -f "$MODEL_DIR/genai_config.json" ]; then
  echo "ERROR: no encuentro genai_config.json en $MODEL_DIR"
  echo "Carpetas descargadas:"
  find models/phi-3.5-cpu -name genai_config.json
  exit 1
fi

echo "== 2/3 Ajustando max_length"
# El valor original (131072) hace que se reserve memoria para todo el contexto
# cuando un generador no fija su propio max_length, y el proceso muere por memoria.
python - <<'EOF'
import json, os
ruta = os.path.join(os.environ["MODEL_DIR"], "genai_config.json")
config = json.load(open(ruta))
antes = config["search"]["max_length"]
config["search"]["max_length"] = 4096
json.dump(config, open(ruta, "w"), indent=4)
print(f"max_length: {antes} -> 4096")
EOF

echo "== 3/3 Prueba de generación"
python scripts/01_base.py

echo
echo "Listo. Ahora prueba: python scripts/02_constrained.py"