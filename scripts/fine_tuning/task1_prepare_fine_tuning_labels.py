"""Convierte el dataset utilizado en ICL/DSPy (CSV) a un formato 
compatible con Fine-tuning"""

from pathlib import Path
import csv
import json
import random

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data" / "task1" / "test"

# Crear directorio de output
output_dir = project_root / "data" / "task1" / "fine_tuning"
output_dir.mkdir(parents=True, exist_ok=True)

rows = []

# Leer el CSV original
with open(data_dir / "test_metadata.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "input": row["sequence"],   # entrada del modelo
            "output": row["label"]      # etiqueta esperada
        })

# Mezclar para evitar sesgos
random.seed(42)
random.shuffle(rows)

# Split train / validation (80 / 20)
split = int(0.8 * len(rows))
train_rows = rows[:split]
val_rows = rows[split:]

def save_dataset(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

save_dataset(output_dir / "train_labels.jsonl", train_rows)
save_dataset(output_dir / "val_labels.jsonl", val_rows)

print("✅ Dataset de fine-tuning por etiquetas")
print(f"   Train: {len(train_rows)} ejemplos")
print(f"   Val:   {len(val_rows)} ejemplos")
print(f"   Ruta:  {output_dir}")
