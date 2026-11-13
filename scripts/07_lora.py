"""Entrena un adaptador LoRA sobre un modelo pequeño con data/train.jsonl, en CPU.

El modelo aprende a producir la respuesta correcta del contrato v2, incluido
«no consta» (null) y la lista de acciones vacía. Guarda el adaptador y el modelo
fusionado en formato Hugging Face, listo para convertir con Olive.

Uso:  python scripts/07_lora.py --prueba            (1 minuto: comprueba que funciona)
      python scripts/07_lora.py                     (entrenamiento completo)
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from contrato_v2 import mensajes, respuesta

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--salida", default="models/qwen-0.5b-lora")
ap.add_argument("--epocas", type=int, default=2)
ap.add_argument("--max-ejemplos", type=int, default=0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--acumular", type=int, default=4)
ap.add_argument("--prueba", action="store_true", help="8 ejemplos, 1 época, no guarda el modelo")
a = ap.parse_args()

torch.manual_seed(13)
random.seed(13)
torch.set_num_threads(os.cpu_count() or 4)

tok = AutoTokenizer.from_pretrained(a.base)
try:
    modelo = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
except TypeError:                      # versiones de transformers anteriores a la 5
    modelo = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)

modelo = get_peft_model(modelo, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
modelo.print_trainable_parameters()

with open("data/train.jsonl", encoding="utf-8") as f:
    docs = [json.loads(linea) for linea in f if linea.strip()]
if a.prueba:
    docs, a.epocas = docs[:8], 1
elif a.max_ejemplos:
    docs = docs[:a.max_ejemplos]


def codificar(doc: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt idéntico al de la evaluación; solo la respuesta cuenta para la pérdida."""
    prompt = tok.apply_chat_template(mensajes(doc["texto"]), tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(respuesta(doc["esperado"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return torch.tensor([p + r]), torch.tensor([[-100] * len(p) + r])


ejemplos = [codificar(d) for d in docs]
print(f"{len(ejemplos)} ejemplos, {a.epocas} época(s), "
      f"longitud media {sum(e[0].shape[1] for e in ejemplos) // len(ejemplos)} tokens", flush=True)

opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=a.lr)
total = len(ejemplos) * a.epocas
modelo.train()
paso, inicio, media = 0, time.time(), 0.0
for epoca in range(a.epocas):
    random.shuffle(ejemplos)
    for ids, etiquetas in ejemplos:
        perdida = modelo(input_ids=ids, labels=etiquetas).loss
        (perdida / a.acumular).backward()
        paso += 1
        media += perdida.item()
        if paso % a.acumular == 0 or paso == total:
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if paso % 20 == 0 or paso == total or a.prueba:
            seg = (time.time() - inicio) / paso
            n = 1 if a.prueba else (paso % 20 or 20)
            print(f"paso {paso}/{total} | pérdida {media / n:.3f} | {seg:.1f} s/paso | "
                  f"faltan {seg * (total - paso) / 60:.0f} min", flush=True)
            media = 0.0

if a.prueba:
    seg = (time.time() - inicio) / paso
    print(f"\nPrueba correcta. El entrenamiento completo (400 ejemplos, 2 épocas) "
          f"tardaría unos {seg * 800 / 60:.0f} minutos.")
else:
    modelo.save_pretrained(a.salida + "-adaptador")
    fusionado = modelo.merge_and_unload()
    fusionado.save_pretrained(a.salida)
    tok.save_pretrained(a.salida)
    print(f"\nAdaptador en {a.salida}-adaptador | modelo fusionado en {a.salida}")