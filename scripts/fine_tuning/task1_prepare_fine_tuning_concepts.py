"""Convierte el dataset utilizado en ICL/DSPy (CSV) a un formato 
compatible con Fine-tuning por conceptos en formato JSONL"""

from pathlib import Path
import csv
import json
import random

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data" / "task1" / "test"

# Crear directorio de output
output_dir = project_root / "data" / "task1" / "fine_tuning"
output_dir.mkdir(parents=True, exist_ok=True)

# Lista para almacenar los ejemplos
rows = []

# Leer el CSV original
with open(data_dir / "test_metadata.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "input": row["sequence"],   # entrada del modelo
            "label": row["label"],      # para análisis
            "concept": int(row["total_picos"])      # concepto esperada
        })

# Mezclar para evitar sesgos
random.seed(42)
random.shuffle(rows)

# Dividir el dataset en entrenamiento (80%) y validación (20%)
split = int(0.8 * len(rows))
train_rows = rows[:split]
val_rows = rows[split:]

# Guardar el JSONL (un ejemplo por línea)
def save_dataset(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

#Guardar el dataset de entrenamiento y validación
save_dataset(output_dir / "train_concepts.jsonl", train_rows)
save_dataset(output_dir / "val_concepts.jsonl", val_rows)

print("✅ Dataset de fine-tuning por conceptos")
print(f"   Train: {len(train_rows)} ejemplos")
print(f"   Val:   {len(val_rows)} ejemplos")
print(f"   Ruta:  {output_dir}")
